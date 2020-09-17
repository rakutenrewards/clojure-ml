(ns curbside.ml.models.xgboost
  "This namespace defines a simple Clojure interface to XGBoost, a library for
   gradient-boosted decision trees. This library gives extremely good
   performance on many problems and is becoming very popular. You can learn more
   about it at [the XGBoost website](https://xgboost.readthedocs.io/en/latest/)."
  (:refer-clojure :exclude [load])
  (:require
   [clojure.java.io :as io]
   [clojure.tools.logging :as log]
   [clojure.set :as set]
   [clojure.spec.alpha :as s]
   [clojure.string :as string]
   [clojure.walk :as walk]
   [curbside.ml.training-sets.conversion :as conversion]
   [curbside.ml.training-sets.sampling :as sampling]
   [curbside.ml.utils.parsing :as parsing])
  (:import
   (ml.dmlc.xgboost4j LabeledPoint)
   (ml.dmlc.xgboost4j.java Booster DMatrix DMatrix$SparseType XGBoost XGBoostError)))

(def default-num-rounds 10)

(defn- ->LabeledPoint
  "Create a labeled point from a vector."
  ([label features]
   (let [non-zeros-indexed (keep-indexed
                            (fn [i v]
                              (when-not (or (nil? v) (zero? v))
                                [i v]))
                            features)]
     (LabeledPoint.
      (float label)
      (int-array (map first non-zeros-indexed))
      (float-array (map second non-zeros-indexed)))))
  ([[label & features]]
   (->LabeledPoint label features)))

(defn- filepath->regression-DMatrix
  [filepath]
  (let [columns-keys (conversion/csv-column-keys filepath)
        vectors (->> filepath
                     (conversion/csv-to-maps)
                     (map #(->LabeledPoint ((apply juxt columns-keys) %))))]
    (DMatrix. (.iterator vectors) nil)))

(defn- ranking-label-keys
  "Given the `column-keys` present in a csv dataset, returns which ones represent
  a label. For ranking tasks, there are multiple labels per example. Each label
  represents the score of a specific document."
  [column-keys]
  (filter (fn [k]
            (string/starts-with? (name k) "label-"))
          column-keys))

(defn- one-hot-vectors
  "Returns a sequence of `n` one-vectors

  [1, 0, 0, ...], [0, 1, 0, ...], [0, 0 , 1, ...]

  of size `n`."
  [n]
  (for [i (range n)]
    (vec (concat (repeat i 0)
                 [1]
                 (repeat (- n i 1) 0)))))

(defn- ranking-example-tuples
  "Given an example map, returns a sequence of tuples [label features]. A tuple is
  generated for each `label-keys`. `features` consists of the concatenation of
  the values of `features-keys` and a one-hot vector."
  [label-keys feature-keys example]
  (let [labels (map #(% example) label-keys)
        features (map #(% example) feature-keys)]
    (map (fn [label one-hot-vector]
           [label (vec (concat features one-hot-vector))])
         labels
         (one-hot-vectors (count labels)))))

(defn- ranking-DMatrix-rows
  [label-keys feature-keys example-maps]
  (mapcat (partial ranking-example-tuples label-keys feature-keys)
          example-maps))

(defn- filepath->ranking-DMatrix
  [filepath]
  (let [example-maps (conversion/csv-to-maps filepath)
        ks (keys (first example-maps))
        label-keys (ranking-label-keys ks)
        feature-keys (set/difference (set ks) (set label-keys))
        points (->> example-maps
                    (ranking-DMatrix-rows label-keys feature-keys)
                    (map (fn [[label features]]
                           (->LabeledPoint label features))))
        groups (repeat (count example-maps) (count label-keys))]
    (doto (DMatrix. (.iterator points) nil)
      (.setGroup (int-array groups))
      (.setWeight (float-array (repeat (count groups) 1.0))))))

(defn- filepath->DMatrix
  [predictor-type filepath]
  (if (= predictor-type :ranking)
    (filepath->ranking-DMatrix filepath)
    (filepath->regression-DMatrix filepath)))

(defn- set-sample-weights!
  "Sets the sample weights vector of a DMatrix. Mutates the DMatrix and returns
   nil."
  [dmatrix weights]
  (assert (= (count weights) (.rowNum dmatrix)))
  (.setWeight dmatrix (float-array weights)))

(defn- split-DMatrix
  "Split the end of a DMatrix off into a second DMatrix.
   Returns the two parts of the DMatrix.

   E.g. (split-DMatrix m 0.2) returns [m1 m2], where m2 is the last 20% of
   the dataset."
  [m split-portion]
  (let [n (.rowNum m)
        split-count (- n (Math/floor (* split-portion n)))
        first-indices (int-array (range 0 split-count))
        last-indices (int-array (range split-count n))]
    [(.slice m first-indices) (.slice m last-indices)]))

(defn train
  ([predictor-type training-set-path hyperparams]
   (train predictor-type training-set-path hyperparams nil))
  ([predictor-type training-set-path
    {:keys [early-stopping-rounds num-rounds validation-set-size]
     :as hyperparameters}
    example-weights-path]
   (let [dm (filepath->DMatrix predictor-type training-set-path)
         weights (when example-weights-path (sampling/filepath->sample-weights example-weights-path))
         _ (when weights (set-sample-weights! dm weights))
         [train val] (if (some? validation-set-size)
                       (split-DMatrix dm validation-set-size)
                       [dm nil])
         booster (or (:booster hyperparameters) "gbtree")
         model (XGBoost/train
                train
                (walk/stringify-keys hyperparameters)
                (int (or num-rounds default-num-rounds))
                (if validation-set-size {"validation" val} {})
                nil
                nil
                nil
                (or early-stopping-rounds 0))]
     (.setAttr model "booster" booster)
     {:xgboost-model model
      :booster booster})))

(defn dispose
  "Frees the memory allocated to given model."
  [model]
  (locking model
    (.dispose (:xgboost-model model))))

(defn- get-xgboost-handle
  "Gets the internal handle field that points to the underlying C++ Booster
   object."
  [^Booster obj]
  (let [m (.. obj getClass (getDeclaredField "handle"))]
    (. m (setAccessible true))
    (. m (get obj))))

(defn- ->predict-DMatrix
  "Convert a 1D vec of floats into an DMatrix meant for use as an input to a
  Booster's .predict() method."
  [features]
  (DMatrix. (.iterator [(->LabeledPoint 1.0 ;; The 1.0 label will be ignored when doing prediction
                                        features)]) nil))

(defn predict
  [{:keys [xgboost-model booster] :as _model} hyperparameters feature-vector]
  (let [dmatrix (->predict-DMatrix feature-vector)]
    (->
     ;; lock for mutual exclusion w.r.t. dispose.
     (locking xgboost-model
       ;; hack: most xgboost code paths check that handle is not null and throw
       ;; an error, but sometimes calling predict just segfaults when the
       ;; handle is a null pointer.
       (if (= 0 (get-xgboost-handle xgboost-model))
         (throw (XGBoostError. "already disposed."))
         (if (= booster "dart")
           ;; Pass a large integer to use all available trees in the model
           (.predict xgboost-model dmatrix false Integer/MAX_VALUE)
           (.predict xgboost-model dmatrix))))
     (ffirst))))

(defn iff
  [& args]
  (or (every? identity args)
      (every? not args)))

(s/def ::double-between-zero-and-one (s/double-in :min 0.0 :max 1.0))
(s/def ::positive-double (s/double-in :min 0.0 :infinite? false))

(s/def ::booster #{"gbtree" "gblinear" "dart"})
(s/def ::silent (s/int-in 0 2))
(s/def ::nthread integer?)
(s/def ::learning_rate ::double-between-zero-and-one)
(s/def ::gamma ::positive-double)
(s/def ::max_delta_step ::positive-double)
(s/def ::max_depth integer?)
(s/def ::min_child_weight ::double-between-zero-and-one)
(s/def ::subsample ::double-between-zero-and-one)
(s/def ::colsample_bytree ::double-between-zero-and-one)
(s/def ::colsample_bylevel ::double-between-zero-and-one)
(s/def ::lambda ::double-between-zero-and-one)
(s/def ::alpha ::double-between-zero-and-one)
(s/def ::tree_method #{"auto" "exact" "approx" "hist" "gpu_exact" "gpu_hist"})
(s/def ::sketch_eps ::double-between-zero-and-one)
(s/def ::scale_pos_weight ::double-between-zero-and-one)
(s/def ::updater
  #{"grow_colmaker"
    "distcol"
    "grow_histmaker"
    "grow_local_histmaker"
    "grow_skmaker"
    "sync"
    "refresh"
    "prune"})
(s/def ::refresh_leaf (s/int-in 0 2))
(s/def ::process_type #{"default" "update"})
(s/def ::grow_policy #{"depthwise" "lossguide"})
(s/def ::max_leaves integer?)
(s/def ::max_bin integer?)
(s/def ::predictor #{"cpu_predictor" "gpu_predictor"})
(s/def ::sample_type #{"uniform" "weighted"})
(s/def ::normalize_type #{"tree" "forest"})
(s/def ::rate_drop ::double-between-zero-and-one)
(s/def ::one_drop (s/int-in 0 2))
(s/def ::skip_drop ::double-between-zero-and-one)
(s/def ::updater #{"shotgun" "coord_descent"})
(s/def ::tweedie_variance_power ::double-between-zero-and-one)
(s/def ::objective #{"reg:logistic"
                     "binary:logistic"
                     "binary:logitraw"
                     "binary:hinge"
                     "gpu:reg:linear"
                     "gpu:reg:logistic"
                     "gpu:binary:logistic"
                     "gpu:binary:logitraw"
                     "count:poisson"
                     "survival:cox"
                     "multi:softmax"
                     "multi:softprob"
                     "rank:ndcg"
                     "rank:pairwise"
                     "reg:gamma"
                     "reg:tweedie"
                     "reg:squarederror"
                     "reg:squaredlogerror"})
(s/def ::base_score (s/double-in :infinite? false :NaN? false))
(s/def ::seed integer?)
(s/def ::num-rounds integer?)
(s/def ::validation-set-size ::double-between-zero-and-one)
(s/def ::early-stopping-rounds integer?)
(s/def ::weight-mean (s/double-in :infinite? false :NaN? false))
(s/def ::weight-label-name (s/or :kw keyword? :str string?))
(s/def ::weight-stddev (s/double-in :infinite? false :NaN? false))

(s/def ::hyperparameters
  (s/and
   (s/keys :req-un [::num-rounds]
           :opt-un [::booster
                    ::silent
                    ::nthread
                    ::learning_rate
                    ::gamma
                    ::max_depth
                    ::min_child_weight
                    ::max_delta_step
                    ::subsample
                    ::colsample_bytree
                    ::colsample_bylevel
                    ::lambda
                    ::alpha
                    ::tree_method
                    ::sketch_eps
                    ::scale_pos_weight
                    ::updater
                    ::refresh_leaf
                    ::process_type
                    ::grow_policy
                    ::max_leaves
                    ::max_bin
                    ::predictor
                    ::sample_type
                    ::normalize_type
                    ::rate_drop
                    ::one_drop
                    ::skip_drop
                    ::updater
                    ::tweedie_variance_power
                    ::objective
                    ::base_score
                    ::seed
                    ::validation-set-size
                    ::early-stopping-rounds
                    ::weight-mean
                    ::weight-label-name
                    ::weight-stddev])
   #(iff (:validation-set-size %) (:early-stopping-rounds %))
   #(iff (:weight-mean %) (:weight-label-name %) (:weight-stddev %))))

(defn save
  [{:keys [xgboost-model] :as _model} filepath]
  (.saveModel xgboost-model filepath)
  [filepath])

(defn- get-booster-from-attributes
  [xgboost-model]
  (.getAttr xgboost-model "booster"))

(defn- ^:deprecated get-booster-from-file
  [filepath]
  (log/info "[xgboost] DEPRECATED Loading the booster type from the binary file")
  (let [booster (->> (io/reader filepath)
                     (line-seq)
                     (keep #(re-find #"dart|gbtree" %))
                     (first))]
    (if (some? booster)
      booster
      (do
        (log/error "[xgboost] Could not find the booster type in the file. Assuming gbtree. This can lead to undefined behaviors if it was a dart booster.")
        "gbtree"))))

(defn load
  [filepath]
  ;; On extremely rare occasions, the model loading can silently fail, resulting
  ;; in a 0 model handle.
  {:post [(not= 0 (get-xgboost-handle (:xgboost-model %)))]}
  (let [m (XGBoost/loadModel ^String filepath)]
    {:xgboost-model m
     :booster (or (get-booster-from-attributes m)
                  (get-booster-from-file filepath))}))
