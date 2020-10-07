(ns curbside.ml.data.scaling
  "Feature scaling is a method used to standardize the range of independent
   variables or features of data. Depending on the training algorithm (like
   SVM), one of the most important task of the creation of the training set is
   to normalize the feature's values to help the algorithm to learn.

   We support two types of scaling:
   - feature scaling :: as its name suggests, it is scaling applied to the input
     features of a model
   - label scaling :: scaling applied to only the =:label= key of a feature map.

   The key distinction between the two types of scaling is that label scaling
   *also needs to be applied on predicted values at inference time*. The scaling
   configuration spec is detailed [here](file:~/curbside-prediction/org/src/curbside/prediction/pipeline.org::*Scale%20Training%20Sets).
"
  (:require
   [clojure.spec.alpha :as s]
   [curbside.ml.data.conversion :as conversion]
   [medley.core :as medley]))

(defmulti compute-factors
  "Given a scaling function, `compute-factors` produces the map of factor
  parameters passed to each `apply-scaling` and `apply-uncaling` call.

  An example factors data structure:

  {:features [{:x {:min 10
                   :max 12}
               :y {:min 0
                   :max 20}}]
   :labels [{:min 10 :max 30}]

   =factors= is a map which contains two keys:
   - =:features= :: Factors of the =:feature-scaling-fns=. It is a vector of
     maps, where each map contains the factors of a given scaling functions.
     For each =:feature-scaling-fns=, the key represents the feature which is
     scaled, and the value is the factors used to scaled this feature.
   - =:labels= :: Factors of the =:label-scaling-fns=. It is a vector of maps.
     Unlike the =:features= key, there is only one value scale, namely the
     label."
  (fn [scaling-fn _dataset]
    scaling-fn))

(defmulti apply-scaling
  "Scales a `value` according to the provided `factors`."
  (fn [scaling-fn _value _value-factors]
    scaling-fn))

(defmulti apply-unscaling
  "Unscales a `value` according to the provided `factors`."
  (fn [scaling-fn _value _value-factors]
    scaling-fn))

(defn scale-map-keys
  [scaling-fn map factors]
  (medley/map-kv-vals (fn [key val]
                        (if-let [key-factors (get factors key)]
                          (when val (apply-scaling scaling-fn val key-factors))
                          val))
                      map))

(s/def ::min number?)
(s/def ::max number?)
(s/def ::min-max-factors (s/and (s/keys :req-un [::min ::max])
                                (fn [{:keys [min max]}]
                                  (< min max))))
(s/def ::log10-factors map?)

(s/def ::value-factors (s/or :min-max ::min-max-factors
                             :log10 ::log10-factors))

(s/def ::features (s/coll-of (s/map-of keyword? ::value-factors)))
(s/def ::labels (s/coll-of ::value-factors))
(s/def ::dataset-factors (s/keys :req-un [::features ::labels]))

(defn- min-max-feature
  [feature dataset]
  (let [values (keep feature dataset)]
    (if (empty? values)
      {:min Double/MIN_VALUE :max Double/MAX_VALUE}
      {:min (apply min values) :max (apply max values)})))

(defmethod compute-factors :min-max
  [_ dataset]
  (let [features (remove #(= :label %) (keys (first dataset)))]
    (reduce (fn [factors feature]
              (assoc factors feature (min-max-feature feature dataset)))
            {}
            features)))

(defmethod apply-scaling :min-max
  [_ value {:keys [min max] :as factors}]
  {:pre [(s/valid? ::min-max-factors factors)]}
  (let [denom (- max min)]
    (/ (- value min)
       (if (> denom 0) denom 1))))

(defmethod apply-unscaling :min-max
  [_ value {:keys [min max] :as factors}]
  {:pre [(s/valid? ::min-max-factors factors)]}
  (+ min
     (* value (- max min))))

(def min-log10-value 1e-8)
(def max-log10-value 1e8)

;; This scaling function applies a =log10= operation to the label. This has been
;; shown in research that this scaling function gives more importance to small
;; label values, improving the overall MAPE metrics.

;; To avoid the =Infinity= values in computation, the minimum value produced by
;; log10 scaling if =1e-8= and the maximum value produced by the unscaling is
;; =1e8=


(defmethod compute-factors :log10
  [& _args]
  {}) ;; Empty map, no factors needs to be saved for this scaling function

(defmethod apply-scaling :log10
  [_ value _value-factors]
  (max min-log10-value (Math/log10 (max 0 value))))

(defmethod apply-unscaling :log10
  [_ value _value-factors]
  (min max-log10-value (Math/pow 10 value)))

(defn scale-feature-map
  "Scales a `feature-map`, applying in order the `scaling-fns` to all features
  present in the `factors` map."
  [scaling-fns factors feature-map]
  (reduce (fn [feature-map [scaling-fn factors]]
            (scale-map-keys scaling-fn feature-map factors))
          feature-map
          (map vector scaling-fns (:features factors))))

(defn scale-dataset-features
  "Scales the features a training set, which is a collection of feature maps."
  [scaling-fns factors dataset]
  (map (partial scale-feature-map scaling-fns factors) dataset))

(defn scale-dataset-labels
  "Scales the `:label` key of the all the feature maps, successively applying
  the `scaling-fns`."
  [scaling-fns factors dataset]
  (reduce (fn [dataset [scaling-fn factors]]
            (map #(update % :label (partial apply-scaling scaling-fn) factors) dataset))
          dataset
          (map vector scaling-fns (:labels factors))))

(defn scale-dataset
  "Scales the features and the labels of a training set."
  [feature-scaling-fns label-scaling-fns factors dataset]
  {:pre [(s/valid? ::dataset-factors factors)]}
  (->> dataset
       (scale-dataset-features feature-scaling-fns factors)
       (scale-dataset-labels label-scaling-fns factors)))

(defn- scaling-factors
  "Compute the `factors` used to scale a training set."
  [feature-scaling-fns label-scaling-fns dataset]
  (letfn [(compute-all-factors [scaling-fns]
            (mapv #(compute-factors % dataset) scaling-fns))]
    {:features (compute-all-factors feature-scaling-fns)
     :labels (compute-all-factors label-scaling-fns)}))

(defn scale-dataset-csv
  "Scales a training set encoded in the file at `input-csv-path`. The scaled set
  is outputted at `outpt-csv-file`, and the scaling factors used to perform
  scaling are saved at `edn-factors-path`."
  [input-csv-path output-csv-file edn-factors-path feature-scaling-fns label-scaling-fns]
  (let [dataset        (conversion/csv-to-maps input-csv-path)
        factors             (scaling-factors feature-scaling-fns label-scaling-fns dataset)
        scaled-set          (scale-dataset feature-scaling-fns label-scaling-fns factors dataset)]
    (spit edn-factors-path (pr-str factors))
    (conversion/maps-to-csv output-csv-file
                            (conversion/csv-column-keys input-csv-path)
                            scaled-set)))

(defn unscale-label
  "Unscaled a single value, successively unscaling the `scaling-fns` in reverse
   order.

   At inference time, we need to unscale the predicted value to obtain a
   prediction that is in the source domain."
  [scaling-fns factors value]
  {:pre [(s/valid? ::dataset-factors factors)]}
  (reduce (fn [value [scaling-fn factors]]
            (apply-unscaling scaling-fn value factors))
          value
          (reverse (map vector scaling-fns (:labels factors)))))
