(ns curbside.ml.training-sets.sampling
  "Sampling a training set is the action of changing a training set in a way that
   we change the distribution of the classes, or we normalize it by adding or
   removing instances of the training set.

   Sampling is a step that you will perform for classification tasks (not
   regression) normally to make sure that your training set are properly balanced.

   Unbalanced classes can create two problems:

   1. The accuracy (i.e. ratio of test samples for which we predicted the correct
      class) is no longer a good measure of the model performance
   2. The training process might arrive at a local optimum

   There are generally 5 methods to help coping with this situation:

   1. Collect more data
   2. Create copies of training samples
   3. Create augmented copies of training samples
   4. Remove training samples
   5. Train for sensitivity and specificity "
  (:require
   [clojure.math.numeric-tower :as math :refer [expt sqrt]]
   [curbside.ml.training-sets.training-set :as training-set]
   [curbside.ml.utils.spec :as spec-utils])
  (:import
   (weka.filters Filter)))

(def default-sampling-config
  {:without-replacement false
   :bias-to-uniform-class 0.0
   :sample-size-percent 100.0
   :max-sample-size nil})

(defn- config->sample-size
  [dataset-size {:keys [sample-size-percent max-sample-size]}]
  (let [n (int (* dataset-size (/ sample-size-percent 100.0)))]
    (if max-sample-size
      (min max-sample-size n)
      n)))

(defn- sample-no-replacement
  [xs n]
  (take n (shuffle xs)))

(defn- sample-with-replacement
  [xs n]
  (let [xsvec (vec xs)]
    (take n (repeatedly #(rand-nth xsvec)))))

(defn sample-training-set
  [training-set config]
  {:pre [(spec-utils/check ::training-set/training-set training-set)]
   :post [(spec-utils/check ::training-set/training-set %)]}

  ;; TODO: support :bias-to-uniform-class config setting https://github.com/RakutenReady/Team/issues/52375
  (when (some? (:bias-to-uniform-class config))
    (throw (UnsupportedOperationException. "bias to uniform class not yet implemented.")))

  (let [config (merge default-sampling-config config)
        n-values (if (some? (:groups training-set))
                   (count (:groups training-set))
                   (count (:feature-maps training-set)))
        sample-size (config->sample-size n-values config)
        selected-indices (if (:without-replacement config)
                           (sample-no-replacement (range n-values) sample-size)
                           (sample-with-replacement (range n-values) sample-size))]
    (if (some? (:groups training-set))
      (training-set/select-groups training-set selected-indices)
      (training-set/select-examples training-set selected-indices))))

;; Sample weighting is related to sampling, and sets the relative importance of
;; individual rows of the training set based on their numerical attributes.

;; We implement sample weighting based on the Gaussian PDF. This requires a
;; couple parameters: the feature to use to weight the features, the expected
;; average value of that feature, and the standard deviation. Rows with values
;; of the feature near the mean will be weighted more heavily, while those
;; farther from the mean will be weighted less, proportionally to the Gaussian
;; curve.

;; The sample weighting function here loads a CSV training set, computes the
;; sample weights, and returns them as a sequence of floats. This sequence of
;; floats must be passed into a model training function that supports it.
;; Currently, this support only exists for XGBoost.

(defn gaussian-pdf
  "Helper function for creating Gaussian PDF sample weights."
  [y mean stddev]
  (let [exponent (* -0.5 (expt (/ (- y mean) stddev) 2))
        coeff (/ 1 (* stddev (sqrt (* 2 Math/PI))))]
    (* coeff (expt Math/E exponent))))

(defn maps->sample-weights
  "Given a training set where each row is represented as a map,
   return a sample weight vector."
  [maps mean label-feature-name stddev]
  (map
   (fn [s]
     (gaussian-pdf ((keyword label-feature-name) s)
                   mean
                   stddev))
   maps))
