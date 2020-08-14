(ns curbside.ml.models.xgboost
  "This namespace defines a simple Clojure interface to XGBoost, a library for
   gradient-boosted decision trees. This library gives extremely good
   performance on many problems and is becoming very popular. You can learn more
   about it at [the XGBoost website](https://xgboost.readthedocs.io/en/latest/)."
  (:refer-clojure :exclude [load])
  (:require
   [clojure.java.io :as io]
   [clojure.tools.logging :as log]
   [clojure.spec.alpha :as s]
   [clojure.string :as string]
   [clojure.walk :as walk]
   [curbside.ml.training-sets.conversion :as conversion]
   [curbside.ml.training-sets.sampling :as sampling]
   [curbside.ml.utils.parsing :as parsing])
  (:import
   (ml.dmlc.xgboost4j LabeledPoint)
   (ml.dmlc.xgboost4j.java Booster DMatrix XGBoost XGBoostError)))

(def default-num-rounds 10)

(defn- get-non-nil-values-indices
  "Get a vector of non-nil indices from a vector `v`"
  [vec]
  (->> vec
       (keep-indexed (fn [i v]
                       (when v
                         i)))
       int-array))

(defn- ->LabeledPoint
  "Create a labeled point from a vector. The vector can be dense or sparse."
  [v]
  (LabeledPoint.
   (->> v
        first
        parsing/parse-float)
   (->> v
        rest
        (map parsing/parse-float)
        get-non-nil-values-indices)
   (->> v
        rest
        (keep parsing/parse-float)
        float-array)))

(defn- filepath->DMatrix
  [filepath]
  (let [columns-keys (conversion/csv-column-keys filepath)
        vectors (->> filepath
                     (conversion/csv-to-maps)
                     (map #(->LabeledPoint ((apply juxt columns-keys) %))))]
    (DMatrix. (.iterator vectors) nil)))

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
  ([training-set-path hyperparams]
   (train training-set-path hyperparams nil))
  ([training-set-path
    {:keys [early-stopping-rounds num-rounds validation-set-size]
     :as hyperparameters}
    example-weights-path]
   (let [dm (filepath->DMatrix training-set-path)
         weights (when example-weights-path (sampling/filepath->sample-weights example-weights-path))
         _ (when weights (set-sample-weights! dm weights))
         [train val] (split-DMatrix dm (or validation-set-size 0.0))
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
  [vec]
  (DMatrix. (.iterator [(->LabeledPoint vec)]) nil))

(defn predict
  [{:keys [xgboost-model booster] :as _model} hyperparameters feature-vector]
  (let [;; Pad to add a dummy label at the front of the vector.
        ;; It will be ignored when doing prediction
        dmatrix (->predict-DMatrix (into [1.0] feature-vector))]
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
