(ns curbside.ml.models.decision-trees
  "The decision tree classifier (which is implement with multiple different
   algorithms) supports binary and multi-class classification and regression.
   The classifier uses the Weka library.

   The decision tree models namespace includes all the function that manage the
   creation, modifications, saving, loading, deletion and optimization
   specifically related to decision tree models.

   # Supported Hyperparameters
   ## C4.5

   | hyper-parameter                  | description                                                                    | value type | possible values | default |
   |----------------------------------+--------------------------------------------------------------------------------+------------+-----------------+---------|
   | =u=                              | Use unpruned tree                                                              | =boolean=  | = [true, false] = |         |
   | =o=                              | Do not collapse tree                                                           | =boolean=  | = [true, false] = |         |
   | =c=                              | Set confidence threshold for pruning.                                          | =decimal=  | = [0.0, ...] =    |    0.25 |
   | =m=                              | Set minimum number of instances per leaf                                       | =integer=  | = [1, ...] =      |       2 |
   | =r=                              | Use reduced error pruning                                                      | =boolean=  | = [true, false] = |         |
   | =n=                              | Set number of folds for reduced error pruning. One fold is used as pruning set | =integer=  | = [1, ...] =      |       3 |
   | =b=                              | Use binary splits only                                                         | =boolean=  | = [true, false] = |         |
   | =s=                              | Don't perform subtree raising                                                  | =boolean=  | = [true, false] = |         |
   | =l=                              | Do not clean up after the tree has been built                                  | =boolean=  | = [true, false] = |         |
   | =a=                              | Laplace smoothing for predicted probabilities                                  | =boolean=  | = [true, false] = |         |
   | =j=                              | Do not use MDL correction for info gain on numeric attributes                  | =boolean=  | = [true, false] = |         |
   | =q=                              | Seed for random data shuffling                                                 | =integer=  | = [1, ...] =      |       1 |
   | =doNotMakeSplitPointActualValue= | Do not make split point actual value                                           | =boolean=  | = [true, false] = |         |

   ## M5P

   | hyper-parameter | description                                                          | value type | possible values | default |
   |-----------------+----------------------------------------------------------------------+------------+-----------------+---------|
   | =n=             | Use unpruned tree/rules                                              | =boolean=  | = [true, false] = |         |
   | =u=             | Use unsmoothed predictions                                           | =boolean=  | = [true, false] = |         |
   | =r=             | Build regression tree/rule rather than a model tree/rule             | =boolean=  | = [true, false] = |         |
   | =m=             | Set minimum number of instances per leaf                             | =integer=  | = [1, ...] =      |       4 |
   | =l=             | Save instances at the nodes in the tree (for visualization purposes) | =boolean=  | = [true, false] = |         |

   ## Random Forest

   | hyper-parameter                           | description                                                                                         | value type | possible values | default |
   |-------------------------------------------+-----------------------------------------------------------------------------------------------------+------------+-----------------+---------|
   | =P=                                       | Size of each bag, as a percentage of the training set size.                                         | =integer=  | =[1, ...]=      |     100 |
   | =O=                                       | Calculate the out of bag error.                                                                     | =boolean=  | =[true, false]= |         |
   | =store-out-of-bag-predictions=            | Whether to store out of bag predictions in internal evaluation object.                              | =boolean=  | =[true, false]= |         |
   | =output-out-of-bag-complexity-statistics= | Whether to output complexity-based statistics when out-of-bag evaluation is performed.              | =boolean=  | =[true, false]= |         |
   | =attribute-importance=                    | Compute and output attribute importance (mean impurity decrease method)                             | =boolean=  | =[true, false]= |         |
   | =I=                                       | Number of iterations.                                                                               | =integer=  | =[1, ...]=      |     100 |
   | =num-slots=                               | Number of execution slots. (default 1 - i.e. no parallelism) (use 0 to auto-detect number of cores) | =integer=  | =[0, ...]=      |       1 |
   | =K=                                       | Number of attributes to randomly investigate. (<1 = int(log_2(#predictors)+1))                      | =integer=  | =[0, ...]=      |       0 |
   | =M=                                       | Set minimum number of instances per leaf.                                                           | =integer=  | =[1, ...]=      |       1 |
   | =V=                                       | Set minimum numeric class variance proportion of train variance for split.                          | =double=   | =[0.0, ...]=    |    1e-3 |
   | =S=                                       | Seed for random number generator.                                                                   | =integer=  | =[1, ...]=      |       1 |
   | =depth=                                   | The maximum depth of the tree, 0 for unlimited.                                                     | =integer=  | =[0, ...]=      |       0 |
   | =N=                                       | Number of folds for backfitting (default 0, no backfitting).                                        | =integer=  | =[1, ...]=      |       0 |
   | =U=                                       | Allow unclassified instances.                                                                       | =boolean=  | =[true, false]= |         |
   | =B=                                       | Break ties randomly when several attributes look equally good.                                      | =boolean=  | =[true, false]= |         |
   | =do-not-check-capabilities=               | If set, classifier capabilities are not checked before classifier is built (use with caution).      | =boolean=  | =[true, false]= |         |
   | =num-decimal-places=                      | The number of decimal places for the output of numbers in the model.                                | =integer=  | =[1, ...]=      |       2 |
   | =batch-size=                              | The desired batch size for batch prediction.                                                        | =integer=  | =[1, ...]=      |     100 |"
  (:refer-clojure :exclude [load])
  (:require
   [clojure.java.io :as io]
   [clojure.spec.alpha :as s]
   [clojure.string :as str]
   [curbside.ml.data.conversion :as conversion]
   [curbside.ml.utils.weka :as weka])
  (:import
   (guru.nidi.graphviz.engine Format Graphviz)
   (guru.nidi.graphviz.parse Parser)
   (java.io BufferedInputStream FileInputStream FileOutputStream ObjectInputStream ObjectOutputStream)
   (weka.classifiers AbstractClassifier)
   (weka.classifiers.trees J48 M5P RandomForest)
   (weka.core Utils)))

(s/def ::u boolean?)
(s/def ::o boolean?)
(s/def ::c (s/double-in :infinite? false :NaN? false))
(s/def ::m integer?)
(s/def ::r boolean?)
(s/def ::n integer?)
(s/def ::b boolean?)
(s/def ::s boolean?)
(s/def ::l boolean?)
(s/def ::a boolean?)
(s/def ::j boolean?)
(s/def ::q integer?)

(s/def ::c45-hyperparameters (s/keys :opt-un [::u
                                              ::o
                                              ::c
                                              ::m
                                              ::r
                                              ::n
                                              ::b
                                              ::s
                                              ::l
                                              ::a
                                              ::j
                                              ::q]))

(s/def ::m5p-hyperparameters (s/keys :opt-un [::u
                                              ::o
                                              ::c
                                              ::m
                                              ::r]))

(s/def ::k integer?)
(s/def ::i integer?)
(s/def ::depth integer?)

(s/def ::rf-hyperparameters (s/keys :opt-un [::k
                                             ::i
                                             ::depth]))

(def default-params {})

(defn- serialize-options
  "Create a valid string of options to feed to the different decision tree
  algorithms. `options` is a map where the key is the option's name and the
  value the option's value."
  [options]
  (->> options
       (mapv (fn [[option v]]
               (let [option (if (= (count (name option)) 1)
                              (str/upper-case (name option))
                              (name option))]
                 (if (boolean? v)
                   (str "-" option " ")
                   (str "-" option " " v " ")))))
       (apply str)
       Utils/splitOptions))

(defn- parameters
  "Define all the hyperparameters required by a Decision Tree trainer. Returns the
  serialized hyperparameters."
  [hyperparameters]
  (serialize-options (merge default-params hyperparameters)))

(defn train
  "Train a Decision Tree model for a given training set csv with specified hyperparameters"
  [algorithm predictor-type dataset-csv hyperparameters]
  (let [tree (case algorithm
               :c4.5 (J48.)
               :m5p (M5P.)
               :random-forest (RandomForest.))]
    (.setOptions tree (parameters hyperparameters))
    (.buildClassifier tree (conversion/csv-to-arff dataset-csv predictor-type))
    tree))

(defn save
  "Save a decision tree model on the file system. Also save a `dot` representation
  of the decision tree model. Return the list of files that got saved on the
  file system."
  [model filepath]
  (let [graph? (not (instance? RandomForest model))
        dot-file (str filepath ".dot")
        png-file (str filepath ".png")]
    (when graph?
      (spit dot-file (.graph model))
      (try
        (let [dot-graph (Parser/read (io/file dot-file))
              png-renderer (.render (Graphviz/fromGraph dot-graph) Format/PNG)]
          (.toFile png-renderer (io/file png-file)))
        (catch Exception e nil)))
    (with-open [output (-> (io/file filepath)
                           FileOutputStream.
                           ObjectOutputStream.)]
      (.writeObject output model))
    (if graph?
      [filepath dot-file png-file]
      [filepath])))

(defn load
  "Load a decision tree model from the file system into memory"
  [filepath]
  (with-open [inp (-> (io/file filepath)
                      FileInputStream.
                      BufferedInputStream.
                      ObjectInputStream.)]
    (.readObject inp)))

(defn load-from-bytes
  [bytes]
  (with-open [input (io/input-stream bytes)]
    (.readObject (ObjectInputStream. input))))

(defn predict
  ;; There is some complexity inherent to classifying decision tree instances
  ;; due to the nature of the classifier. Depending on the decision tree
  ;; algorithm, it can classify instances that have numeric or nominal
  ;; attributes. The nominal attributes can have numerous possible values.
  ;; Because of the possible complexity of the instances we want to predict, we
  ;; have to carry around all the attributes used to create the training set
  ;; dataset such that we can properly create the instance that we want to
  ;; classify.

  ;; When we want to classify/predict a new instance, we have to use the
  ;; =create-instance= function. That function takes the problem used to
  ;; classify the instance and the =features= that describes the instance.
  ;; The features are a map where the keys are the names/indexes of the attributes.
  ;; If a key is a keyword, then it is converted into a string. The =problem= is
  ;; the one used to create the model. One thing that can be done is simply to
  ;; define the header of a =ARFF= file that you will load with the = (problem) =
  ;; function. The important is to have the header and all the definition of
  ;; each attribute.

  [predictor-type ^AbstractClassifier model selected-features feature-vector]
  (let [instance (weka/create-instance predictor-type selected-features feature-vector)]
    (.classifyInstance model instance)))
