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
  [_ predictor-type training-set-path params & [weights-path]]
  (if weights-path
    (xgboost/train predictor-type training-set-path params weights-path)
    (xgboost/train predictor-type training-set-path params)))

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
       (unscale-label label-scaling-fns scaling-factors)))

(defn- to-temp-csv-path
  [header rows]
  (let [file (doto (File/createTempFile "data_" ".csv")
               (.deleteOnExit))]
    (with-open [w (io/writer file)]
      (csv/write-csv w (concat [header] rows)))
    (.getPath file)))

(defn- classify
  [actual predicted]
  (NominalPrediction. actual (NominalPrediction/makeDistribution predicted 2)))

(defn- evaluate-using-model
  "Given a model and all the details about features/labels, generate evaluation metrics
  for the given labels using the model."
  [model algorithm selected-features validation-set predictor-type hyperparameters
   scaling-factors label-scaling-fns]
  (reduce (fn [agg [abs-error square-error predictions]]
            (-> agg
                (update :abs-error + abs-error)
                (update :square-error + square-error)
                (update :n inc)
                (update :predictions conj predictions)))
          {:abs-error 0
           :square-error 0
           :n 0
           :predictions []}
          (for [[label & features] validation-set]
              (let [features-map (into {} (map vector selected-features features))
                    predicted-value (infer algorithm predictor-type model selected-features hyperparameters features-map
                                           :scaling-factors scaling-factors
                                           :feature-scaling-fns nil ;; The features are already scaled in the training set.
                                           :label-scaling-fns label-scaling-fns)
                    prediction (if (= predictor-type :classification)
                                 (classify (Double/parseDouble label) predicted-value))
                    unscaled-label (unscale-label label-scaling-fns scaling-factors (parsing/parse-double label))
                    diff (- unscaled-label predicted-value)
                    abs-error (Math/abs diff)
                    square-error (* diff diff)]
                [abs-error square-error prediction]))))

(defn- train-and-evaluate
  [algorithm selected-features hyperparameters label-scaling-fns scaling-factors
   training-csv-path training-weights-path validation-set predictor-type]
  (let [model (train algorithm predictor-type training-csv-path hyperparameters training-weights-path)
        evaluation_result (evaluate-using-model model algorithm selected-features validation-set predictor-type hyperparameters
                                                scaling-factors label-scaling-fns)]
    (dispose algorithm model)
    evaluation_result))

(defn- zip
  [xs ys]
  (map vector xs ys))

(defn- unzip
  [xs]
  [(map first xs) (map second xs)])

(defn- load-weights
  "Loads the weights file, if it can be found and is non-empty. If not, return
   constant weights of the same length as the training set."
  [training-set example-weights-path]
  (if (and example-weights-path (.exists (io/file example-weights-path)))
    (with-open [in-file (io/reader example-weights-path)]
      (let [[_header & weights] (csv/read-csv in-file)]
        (if (not= (count weights) (count training-set))
          (throw (Exception. "Weights file is not the same length as training set file!"))
          weights)))
    (repeat (count training-set) ["1.0"])))

(defn- train-test-split
  "Produce the dataset splits between train and validate using the given % split"
  [training-set-path example-weights-path train-split-percentage]
  (let [[header & training-set] (with-open [in-file (io/reader training-set-path)]
                                  (doall
                                   (csv/read-csv in-file)))
        weights (load-weights training-set example-weights-path)
        total-length (count training-set)
        train-set-size (Math/ceil (/ (* total-length train-split-percentage) 100))
        validation-set-size (- total-length train-set-size)
        shuffled-set (shuffle (zip training-set weights))
        training-subset (take train-set-size shuffled-set)
        validation-subset (take-last validation-set-size shuffled-set)
        [validation-set _] (unzip validation-subset)
        training-csv-path (to-temp-csv-path header (map (partial map first) training-subset))
        training-weights-path (to-temp-csv-path ["weight"] (map (partial map second) training-subset))]
    [{:training-weights-path training-weights-path :training-csv-path training-csv-path :validation-set validation-set}]))

(defn- k-fold-split
  "Produce the dataset splits using to do k-fold cross validation"
  [training-set-path example-weights-path k-folds]
  (let [[header & training-set] (with-open [in-file (io/reader training-set-path)]
                                  (doall
                                   (csv/read-csv in-file)))
        weights (load-weights training-set example-weights-path)
        folds (partition-all (/ (count training-set) k-folds) (shuffle (zip training-set weights)))
        evaluation-splits   (loop [processed-folds 1
                                   [validation-set validation-weights] (unzip (first folds))
                                   training-set-folds (rest folds)
                                   result []]
                              (if (<= processed-folds k-folds)
                                (let [training-csv-path (to-temp-csv-path header (apply concat (map (partial map first) training-set-folds)))
                                      training-weights-path (to-temp-csv-path ["weight"] (apply concat (map (partial map second) training-set-folds)))]
                                  (recur (inc processed-folds)
                                         (unzip (first training-set-folds))
                                         (conj (rest training-set-folds) (zip validation-set validation-weights))
                                         (conj result {:training-weights-path training-weights-path :training-csv-path training-csv-path :validation-set validation-set})))
                                result))]
    evaluation-splits))

(defn- create-train-validate-splits
  "Given a dataset and the type of split to be used, produce a vector of train and
  validation sets to be evaluated."
  [evaluate-options training-set-path example-weights-path]
  (case (:type evaluate-options)
    :train-test-split (train-test-split training-set-path example-weights-path (:train-split-percentage evaluate-options))
    :k-fold (k-fold-split training-set-path example-weights-path (:folds evaluate-options))))

(defn evaluate
  "Either cross validation or validation using a held out test set"
  [algorithm predictor-type selected-features hyperparameters training-set-path evaluate-options
   & {:keys [scaling-factors label-scaling-fns example-weights-path]}]
  (let [splits-to-evaluate (create-train-validate-splits evaluate-options training-set-path example-weights-path)
        metrics (->> splits-to-evaluate
                     (pmap (fn [{:keys [training-csv-path training-weights-path validation-set]}]
                             (train-and-evaluate algorithm
                                                 selected-features
                                                 hyperparameters
                                                 label-scaling-fns
                                                 scaling-factors
                                                 training-csv-path
                                                 training-weights-path
                                                 validation-set
                                                 predictor-type)))
                     (doall)
                     (reduce (fn [agg {:keys [abs-error square-error n predictions]}]
                               (-> agg
                                   (update :abs-error + abs-error)
                                   (update :square-error + square-error)
                                   (update :n #(+ % n))
                                   (update :predictions concat predictions)))))]
    (metrics/model-metrics predictor-type metrics)))

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
             label-scaling-fns example-weights-path]}]
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
                                       :example-weights-path example-weights-path)]
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
