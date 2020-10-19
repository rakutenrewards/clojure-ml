(ns curbside.ml.models.linear-svm
  "The linear SVM classifier is geared to handle large-scale linear
   classification with sparse features matrix. It supports logistic regression
   and linear support vector machines. The linear SVM classifier uses the
   [https://www.csie.ntu.edu.tw/~cjlin/liblinear/](LIBLINEAR) library."
  (:refer-clojure :exclude [load])
  (:require
   [clojure.java.io :as io]
   [clojure.spec.alpha :as s]
   [curbside.ml.data.conversion :as conversion])
  (:import
   (de.bwaldvogel.liblinear FeatureNode Linear Model Parameter Problem SolverType)
   (java.io File)))

(s/def ::c (s/double-in :infinite? false :NaN? false))
(s/def ::p (s/double-in :infinite? false :NaN? false))
(s/def ::algorithm #{"l2lr-primal"
                     "l2l2"
                     "l2l2-primal"
                     "l2l1"
                     "multi"
                     "l1l2-primal"
                     "l1lr"
                     "l2lr"})
(s/def ::eps (s/double-in :infinite? false :NaN? false))
(s/def ::target-weight-label integer?)
(s/def ::weight integer?)

(s/def ::hyperparameters (s/keys :opt-un [::c
                                          ::p
                                          ::algorithm
                                          ::eps
                                          ::target-weight-label
                                          ::weight]))

(def default-hyperparameters {:algorithm :l2l2
                              :c 1
                              :eps 0.01
                              :max-iterations 256
                              :p 0.1
                              :weights nil})

(defn- feature-vector->feature-node-array
  [feature-vector]
  (->> feature-vector
       (keep-indexed
        (fn [i x]
          (when (number? x)
            (FeatureNode. (inc i) x))))
       (into-array)))

(defn- dataset->problem
  "Define a problem space from a dataset."
  [{:keys [labels feature-maps features]}]
  (let [problem (Problem.)]
    (set! (.l problem) (count labels))
    (set! (.n problem) (count features))
    (set! (.y problem) (double-array labels))
    (set! (.x problem)
          (->> feature-maps
               (map #(conversion/feature-map-to-vector features %))
               (map feature-vector->feature-node-array)
               (into-array)))
    problem))

(defn- parameters
  "Define all the parameters required by a Linear SVM trainer. The `weight`
  parameter define the weight modifier to apply to each class. It is a map of
  where the keys are the classes labels and where the value is the weight
  modifier to apply to the `c` parameter"
  [hyperparameters]
  (let [{:keys [algorithm c eps max-iterations p weights]} (merge default-hyperparameters
                                                                  hyperparameters)
        parameters (new Parameter (case (keyword algorithm)
                                    :l2lr-primal SolverType/L2R_LR
                                    :l2l2 SolverType/L2R_L2LOSS_SVC_DUAL
                                    :l2l2-primal SolverType/L2R_L2LOSS_SVC
                                    :l2l1 SolverType/L2R_L1LOSS_SVC_DUAL
                                    :multi SolverType/MCSVM_CS
                                    :l1l2-primal SolverType/L1R_L2LOSS_SVC
                                    :l1lr SolverType/L1R_LR
                                    :l2lr SolverType/L2R_LR) c eps max-iterations p)]
    (when-not (nil? weights)
      (.setWeights parameters
                   (double-array (vals weights))
                   (into-array Integer/TYPE (keys weights))))
    parameters))

(defn train
  [dataset hyperparameters]
  (Linear/train (dataset->problem dataset) (parameters hyperparameters)))

(defn save
  [model filepath]
  (with-open [out-file (io/writer filepath)]
    (Linear/saveModel out-file ^Model model)
    [filepath]))

(defn load
  [filepath-or-bytes]
  (with-open [reader (io/reader filepath-or-bytes)]
    (Linear/loadModel reader)))

(defn- create-feature-node
  "Create a FeatureNode at `index` with `value`. If `value` is empty then it
  returns nil otherwise it returns the FeatureNode"
  [index value]
  (when-let [value (if (string? value)
                     (when-not (empty? value)
                       (Double/parseDouble value))
                     value)]
    (new FeatureNode (inc index) value)))

(defn predict
  [model feature-vector]
  (Linear/predict model
                  (->> feature-vector
                       (keep-indexed create-feature-node)
                       into-array)))
