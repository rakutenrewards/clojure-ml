(ns curbside.ml.models
  "All models implement a few common functions:

   1. =save= to persist a trained model to disk.
   2. =load= to load a trained model from disk.
   3. =train= to train the model on a given problem.
   4. =predict= to make a prediction
   5. =dispose= to free allocated memory, if applicable

   We will define multimethods for all of these operations. These multimethods
   will switch based on a keyword specifying the algorithm to use. Using a
   keyword allows us to easily specify the algorithm in the pipeline configs.

   This namespace also includes hyperparameter search functionality.
   The supported hyperparameter search functions are:

   1. Grid Search: we exhaustively try each and every combination possible from
      the given search space. Note that for continuous values, it is still
      required to specify a finite list of values to try.
   2. Random Search: From the given search space, we randomly pick values, the
      search space can consist of integers, decimals and strings. The integer
      and decimal spaces are defined by min (inclusive)
      and max (exclusive) while the string can take a finite set of values
      defined in 'values' provided as a list. The total number of combinations
      tried are defined by 'iteration-count' defined in
      'hyperparameter-search-fn'. This would allow us to explore values inside
      the continous search space which need not be explicitly defined the config
      like a grid search. Also, random search allows us to achieve comparable
      results to grid search much faster due to lesser number of iterations
      (depending on the number of combinations tried in grid search and
      iteration count)."
  (:refer-clojure :exclude [load])
  (:require
   [clojure.data.csv :as csv]
   [clojure.java.io :as io]
   [clojure.math.combinatorics :as combinatorics]
   [clojure.spec.alpha :as s]
   [com.climate.claypoole :as cp]
   [curbside.ml.metrics :as metrics]
   [curbside.ml.models.xgboost :as xgboost]
   [curbside.ml.models.decision-trees :as decision-trees]
   [curbside.ml.models.linear-svm :as linear-svm]
   [curbside.ml.models.svm :as svm]
   [curbside.ml.utils.parsing :as parsing]
   [curbside.ml.utils.spec :as spec]
   [curbside.ml.training-sets.scaling :as scaling]
   [curbside.ml.training-sets.training-set :as training-set]
   [curbside.ml.training-sets.conversion :as conversion])
  (:import
   (java.io File)
   (java.util ArrayList)
   (weka.classifiers.evaluation NominalPrediction)))

(s/def ::algorithm #{:lsvm :svm :c4.5 :random-forest :m5p :xgboost})

(s/def ::predictor-type #{:classification :ranking :regression})

(defmulti save
  (fn [algorithm model filepath]
    algorithm))

(defmulti load
  (fn [algorithm filepath]
    algorithm))

(defmulti train
  (fn [algorithm predictor-type training-set-path hyperparameters & args]
    algorithm))

(defmulti predict
  "Not meant to be called directly. Use =infer= instead."
  (fn [algorithm predictor-type model selected-features hyperparameters feature-vector]
    algorithm))

(defmulti dispose
  (fn [algorithm model]
    algorithm))

(defmethod dispose :default
  [_ _]
  nil)

(defmethod save :xgboost
  [_ model filepath]
  (xgboost/save model filepath))

(defmethod load :xgboost
  [_ filepath]
  (xgboost/load filepath))

(defmethod train :xgboost
  [_ predictor-type training-set-path params & [weights-path groups-path]]
  (xgboost/train (training-set/load-csv-files training-set-path weights-path groups-path)
                 nil ;; TODO support passing a the training-set encoding. This may result in breaking changes in the API
                 params))

(defmethod predict :xgboost
  [_ _predictor-type model _seleted-features hyperparameters feature-vector]
  (xgboost/predict model hyperparameters feature-vector))

(defmethod dispose :xgboost
  [_ model]
  (xgboost/dispose model))

(defmethod save :c4.5
  [_algorithm model filepath]
  (decision-trees/save model filepath))

(defmethod save :m5p
  [_algorithm model filepath]
  (decision-trees/save model filepath))

(defmethod save :random-forest
  [_algorithm model filepath]
  (decision-trees/save model filepath))

(defmethod load :c4.5
  [_ file]
  (decision-trees/load file))

(defmethod load :m5p
  [_ file]
  (decision-trees/load file))

(defmethod load :random-forest
  [_ file]
  (decision-trees/load file))

(defmethod train :c4.5
  [algorithm predictor-type training-set hyperparameters & _]
  (decision-trees/train algorithm predictor-type training-set hyperparameters))

(defmethod train :m5p
  [algorithm predictor-type training-set hyperparameters & _]
  (decision-trees/train algorithm predictor-type training-set hyperparameters))

(defmethod train :random-forest
  [algorithm predictor-type training-set hyperparameters & _]
  (decision-trees/train algorithm predictor-type training-set hyperparameters))

(defmethod predict :c4.5
  [_ predictor-type model selected-features _hyperparameters feature-vector]
  (decision-trees/predict predictor-type model selected-features feature-vector))

(defmethod predict :m5p
  [_ predictor-type model selected-features _hyperparameters feature-vector]
  (decision-trees/predict predictor-type model selected-features feature-vector))

(defmethod predict :random-forest
  [_ predictor-type model selected-features _hyperparameters feature-vector]
  (decision-trees/predict predictor-type model selected-features feature-vector))

(defmethod save :svm
  [_ model filepath]
  (svm/save model filepath))

(defmethod load :svm
  [_ filepath]
  (svm/load filepath))

(defmethod train :svm
  [_ _predictor-type training-set-path hyperparameters & _]
  (svm/train training-set-path hyperparameters))

(defmethod predict :svm
  [_ _predictor-type model seleted-features hyperparameters feature-vector]
  (svm/predict model seleted-features hyperparameters feature-vector))

(defmethod save :lsvm
  [_ model filepath]
  (linear-svm/save model filepath))

(defmethod load :lsvm
  [_ filepath]
  (linear-svm/load filepath))

(defmethod train :lsvm
  [_ _predictor-type training-set-csv-path hyperparameters & _]
  (linear-svm/train training-set-csv-path hyperparameters))

(defmethod predict :lsvm
  [_ _predictor-type model _selected-features _hyperparameters feature-vector]
  (linear-svm/predict model feature-vector))

(defn- parse-feature-map
  [selected-features feature-map]
  (reduce-kv #(assoc % %2 (parsing/parse-double %3))
             {}
             (select-keys feature-map selected-features)))

(defn- feature-scaling
  [feature-scaling-fns scaling-factors feature-map]
  (if feature-scaling-fns
    (scaling/scale-feature-map feature-scaling-fns scaling-factors feature-map)
    feature-map))

(defn- unscale-label
  [label-scaling-fns scaling-factors prediction]
  (if label-scaling-fns
    (scaling/unscale-label label-scaling-fns scaling-factors prediction)
    prediction))

(defn- classify
  [predicted]
  (NominalPrediction. predicted (NominalPrediction/makeDistribution predicted 2)))

(defn infer
  "This function performs the inference steps to perform predictions using a
  single trained model. It includes data preparation and post-processing
  operations required by all models. Such operations include:
  - Feature selection
  - Feature scaling (optional)
  - Querying a model prediction
  - Scaling the output of the model (optional)"
  [algorithm predictor-type model selected-features hyperparameters feature-map
   & {:keys [scaling-factors feature-scaling-fns label-scaling-fns]}]
  (->> feature-map
       (parse-feature-map selected-features)
       (feature-scaling feature-scaling-fns scaling-factors)
       (conversion/feature-map-to-vector selected-features)
       (predict algorithm predictor-type model selected-features hyperparameters)
       (#(if (= predictor-type :classification)
           (classify %)
           %))
       (unscale-label label-scaling-fns scaling-factors)))

(defn- infer-batch
  [algorithm predictor-type model selected-features hyperparameters feature-maps & args]
  (mapv #(apply infer algorithm predictor-type model selected-features hyperparameters % args)
        feature-maps))

(defn- train-and-infer
  [algorithm predictor-type selected-features hyperparameters
   scaling-factors label-scaling-fns training-set validation-set]
  (let [{:keys [training-set-path weights-path groups-path]} (training-set/save-temp-csv-files training-set)
        model (train algorithm predictor-type training-set-path hyperparameters weights-path groups-path)
        predictions (infer-batch algorithm predictor-type model selected-features hyperparameters (:feature-maps validation-set)
                                          :scaling-factors scaling-factors :label-scaling-fns label-scaling-fns)]
    (dispose algorithm model)
    predictions))

(defn- create-train-validate-splits
  "Given a dataset and the type of split to be used, produce a vector of train and
  validation sets to be evaluated."
  [evaluate-options training-set]
  (case (:type evaluate-options)
    :train-test-split [(training-set/train-test-split training-set true (:train-split-percentage evaluate-options))]
    :k-fold (training-set/k-fold-split training-set true (:folds evaluate-options))))

(defn evaluate
  "Either cross validation or validation using a held out test set"
  [algorithm predictor-type selected-features hyperparameters training-set-path evaluate-options
   & {:keys [scaling-factors label-scaling-fns example-weights-path example-groups-path]}]
  (let [training-set (training-set/load-csv-files training-set-path example-weights-path example-groups-path)
        splits-to-evaluate (create-train-validate-splits evaluate-options training-set)
        predictions+labels (->> splits-to-evaluate
                                (pmap (fn [[training-set validation-set]]
                                        (map vector
                                             (train-and-infer algorithm
                                                              predictor-type
                                                              selected-features
                                                              hyperparameters
                                                              scaling-factors
                                                              label-scaling-fns
                                                              training-set
                                                              validation-set)
                                             (:labels validation-set))))
                                (apply concat))]
    (metrics/model-metrics predictor-type
                           (map first predictions+labels)
                           (map second predictions+labels))))

(def supported-evaluate-types #{:train-test-split :k-fold})

(s/def :evaluate/type supported-evaluate-types)

(s/def ::evaluate-common
  (s/keys :req-un [:evaluate/type]))

(defmulti evaluate-spec :type)

(s/def ::train-split-percentage (s/int-in 1 99))

(s/def ::folds (s/int-in 1 Integer/MAX_VALUE))

(defmethod evaluate-spec :train-test-split [_]
  (s/merge ::evaluate-common
           (s/keys :req-un [::train-split-percentage])))

(defmethod evaluate-spec :k-fold [_]
  (s/merge ::evaluate-common
           (s/keys :req-un [::folds])))

(s/def ::evaluate (s/multi-spec evaluate-spec :type))

(def supported-hyperparameter-search-fn-types #{:grid :random})

(s/def :hyperparameter-search-fn/type supported-hyperparameter-search-fn-types)

(s/def ::hyperparameter-search-fn-common
  (s/keys :req-un [:hyperparameter-search-fn/type]))

(defmulti hyperparameter-search-fn :type)

(defmethod hyperparameter-search-fn :grid [_]
  ::hyperparameter-search-fn-common)

(s/def ::iteration-count int?)

(defmethod hyperparameter-search-fn :random [_]
  (s/merge
   ::hyperparameter-search-fn-common
   (s/keys :req-un [::iteration-count])))

(s/def ::hyperparameter-search-fn (s/multi-spec hyperparameter-search-fn :type))

(def supported-random-search-space-dimension-types #{"integer" "decimal" "string"})

(s/def :random-search-space-dimension/type supported-random-search-space-dimension-types)

(s/def ::random-search-space-dimension-common
  (s/keys :req-un [:random-search-space-dimension/type]))

(s/def :string/values (s/coll-of string? :distinct true))

(s/def ::random-search-space-dimension-string (s/keys :req-un [:string/values]))

(s/def :decimal/min (s/double-in :infinite? false :NaN? false))
(s/def :decimal/max (s/double-in :infinite? false :NaN? false))
(s/def ::random-search-space-dimension-decimal (s/and (s/keys :req-un [:decimal/min :decimal/max])
                                                      (fn [{:keys [min max]}] (< min max))))

(s/def :integer/min (s/int-in Integer/MIN_VALUE Integer/MAX_VALUE))
(s/def :integer/max (s/int-in Integer/MIN_VALUE Integer/MAX_VALUE))
(s/def ::random-search-space-dimension-integer (s/and (s/keys :req-un [:integer/min :integer/max])
                                                      (fn [{:keys [min max]}] (< min max))))

(defmulti random-search-space-dimension :type)

(defmethod random-search-space-dimension "integer" [_]
  (s/merge
   ::random-search-space-dimension-integer
   ::random-search-space-dimension-common))

(defmethod random-search-space-dimension "decimal" [_]
  (s/merge
   ::random-search-space-dimension-decimal
   ::random-search-space-dimension-common))

(defmethod random-search-space-dimension "string" [_]
  (s/merge
   ::random-search-space-dimension-string
   ::random-search-space-dimension-common))

(s/def ::random-search-space-dimension (s/multi-spec random-search-space-dimension :type))

(s/def ::hyperparameter-search-random-space-key-check keyword?)

(s/def ::hyperparameter-search-space-random
  (s/map-of ::hyperparameter-search-random-space-key-check ::random-search-space-dimension))

(s/def ::hyperparameter-search-space-grid (s/map-of keyword?
                                                    (s/coll-of (s/or :double (s/double-in :infinite? false :NaN? false)
                                                                     :integer integer?
                                                                     :string string?) :distinct true)))

(s/def ::hyperparameter-search-space (s/or :random ::hyperparameter-search-space-random
                                           :grid ::hyperparameter-search-space-grid))

;; Algorithm specific hyperparameters

(s/def ::hyperparameters (s/or :lsvm ::linear-svm/hyperparameters
                               :svm ::svm/hyperparameters
                               :c45 ::decision-trees/c45-hyperparameters
                               :m5p ::decision-trees/m5p-hyperparameters
                               :rf ::decision-trees/rf-hyperparameters
                               :xgboost ::xgboost/hyperparameters))

(defn- grid-search-combos
  "Given the hyperparameter search space, returns all possible combinations of
  parameters."
  [hyperparameter-search-space]
  (->> (vals hyperparameter-search-space)
       (apply combinatorics/cartesian-product)
       (map #(into {} (map (fn [x y] [x y])
                           (keys hyperparameter-search-space)
                           %)))))

(defn- random-value
  "Generate random values for the given set of parameter constraints which are
   used for random search"
  [{:keys [min max type values]}]
  (case type
    "integer" (+ (rand-int (- max min)) min)
    "decimal" (+ (rand (- max min)) min)
    "string" (rand-nth values)))

(defn- random-search-combos
  "Given the hyperparameter search space, generate a given number of random
  combinations of parameters"
  [iteration-count hyperparameter-search-space]
  (repeatedly iteration-count
              #(into {} (map (fn [[key value]] [key (random-value value)])
                             hyperparameter-search-space))))

(defn optimize-hyperparameters
  "This function is responsible for training a model with the best
  hyperparameters found by the provided `hyperparameter-search-fn`."
  [algorithm predictor-type selected-features hardcoded-hyperparameters hyperparameter-search-fn hyperparameter-search-space training-set-path evaluate-options
   & {:keys [selection-metric threads-pool scaling-factors feature-scaling-fns
             label-scaling-fns example-weights-path example-groups-path]}]
  {:pre [(spec/check ::algorithm algorithm)
         (spec/check ::predictor-type predictor-type)
         (spec/check ::hyperparameters hardcoded-hyperparameters)
         (spec/check (s/nilable ::hyperparameter-search-fn) hyperparameter-search-fn)
         (spec/check ::hyperparameter-search-space hyperparameter-search-space)]}
  (let [hyperparameter-search-fn (or hyperparameter-search-fn {:type :grid})
        selection-metric (or selection-metric :root-mean-square-error)
        thread-count (or threads-pool 1)
        combos (case (:type hyperparameter-search-fn)
                 :grid (grid-search-combos hyperparameter-search-space)
                 :random (random-search-combos (:iteration-count hyperparameter-search-fn)
                                               hyperparameter-search-space))
        eval (fn [hyperparameters-to-optimize]
               (let [hyperparameters (merge hardcoded-hyperparameters hyperparameters-to-optimize)
                     metrics (evaluate algorithm
                                       predictor-type
                                       selected-features
                                       hyperparameters
                                       training-set-path
                                       evaluate-options
                                       :scaling-factors scaling-factors
                                       :feature-scaling-fns feature-scaling-fns
                                       :label-scaling-fns label-scaling-fns
                                       :example-weights-path example-weights-path
                                       :example-groups-path example-groups-path)]
                 {:optimal-params hyperparameters
                  :selected-evaluation (get metrics selection-metric)
                  :model-evaluations metrics}))
        find-best (if (= (metrics/comparator selection-metric) <)
                    min-key
                    max-key)
        evaluated-combos (cp/with-shutdown! [pool thread-count]
                           (->> combos
                                (cp/pmap pool eval)
                                (doall)))
        best-evaluation (apply find-best :selected-evaluation evaluated-combos)]
    best-evaluation))
