(ns curbside.ml.models.svm
  "The SVM supports logistic regression and linear support vector machines. The
   linear SVM classifier uses the
   [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) library."
  (:refer-clojure :exclude [load])
  (:require
   [clojure.data.csv :as csv]
   [clojure.java.io :as io]
   [clojure.spec.alpha :as s]
   [clojure.string :as str]
   [curbside.ml.utils.parsing :as parsing])
  (:import
   (clojure.lang Reflector)
   (libsvm svm_problem svm_node svm_parameter svm)))

(s/def ::kernel-type #{:linear
                       :poly
                       :pre-computed
                       :rbf
                       :sigmoid})

(s/def ::svm-type #{:c-svc
                    :epsilon-svr
                    :nu-svc
                    :nu-svr
                    :one-class})

(s/def ::c (s/double-in :infinite? false :NaN? false))
(s/def ::eps (s/double-in :infinite? false :NaN? false))
(s/def ::coef0 (s/double-in :infinite? false :NaN? false))
(s/def ::degree (s/double-in :infinite? false :NaN? false))
(s/def ::gamma (s/double-in :infinite? false :NaN? false))
(s/def ::nr-weight integer?)
(s/def ::nu (s/double-in :infinite? false :NaN? false))
(s/def ::p (s/double-in :infinite? false :NaN? false))
(s/def ::probability integer?)
(s/def ::shrinking integer?)

(s/def ::hyperparameters (s/keys :opt-un [::kernel-type
                                          ::svm-type
                                          ::c
                                          ::coef0
                                          ::degree
                                          ::eps
                                          ::gamma
                                          ::nr-weight
                                          ::nu
                                          ::p
                                          ::probability
                                          ::shrinking]))

(def kernel-types
  {:linear svm_parameter/LINEAR
   :poly svm_parameter/POLY
   :pre-computed svm_parameter/PRECOMPUTED
   :rbf svm_parameter/RBF
   :sigmoid svm_parameter/SIGMOID})

(def svm-types
  {:c-svc svm_parameter/C_SVC
   :epsilon-svr svm_parameter/EPSILON_SVR
   :nu-svc svm_parameter/NU_SVC
   :nu-svr svm_parameter/NU_SVR
   :one-class svm_parameter/ONE_CLASS})

(def default-hyperparameters {:kernel-type (:rbf kernel-types)
                              :svm-type (:c-svc svm-types)
                              :C 1               ; for c-svc, epsilon-svr and nu-svr
                              :cache-size 100    ; in MB
                              :coef0 0           ; for poly and sigmoid
                              :degree 3          ; for poly
                              :eps 1e-3          ; stopping criteria
                              :gamma 0           ; for poly, rbf and sigmoid
                              :nr-weight 0       ; for c-svc
                              :nu 0.5            ; for nu-svc, one-class and nu-svr
                              :p 0.1             ; for epsilon-svr
                              :probability 0     ; do probability estimates
                              :shrinking 1       ; use the shrinking heuristic
                              :weight (double-array 0) ; for c-svc
                              :weight-label (int-array 0)}) ; for c-svc

(defn- training-set-path->problem
  "Define a problem space by reading a CSV training set. If training is a CSV
  file then the problem will be read from that file. If training is a sequence
  that define the problem, then the problem will be created from that sequence."
  [training-set-path]
  (let [training-set (if (and (string? training-set-path)
                              (.exists (io/as-file training-set-path)))
                       (rest
                        (with-open [in-file (io/reader training-set-path)]
                          (doall (csv/read-csv in-file))))
                       (throw (Exception. (str "File doesn't exist: " training-set-path))))
        problem (new svm_problem)]
    (set! (.l problem) (count training-set))
    (set! (.y problem) (double-array (->> training-set
                                          (mapv (fn [[label & _features]]
                                                  (parsing/parse-double label))))))
    (set! (.x problem) (into-array
                        (->> training-set
                             (map (fn [[_label & features]]
                                    (into-array
                                     (sort-by #(.index %)
                                              (->> features
                                                   (keep-indexed (fn [index feature]
                                                                   (when-let [feature (parsing/parse-double feature)]
                                                                     (let [node (new svm_node)]
                                                                       (set! (. node index) (inc index))
                                                                       (set! (. node value) feature)
                                                                       node))))))))))))
    problem))

(defn- format-hyperparameters
  "Define all the hyperparameters required by a SVM trainer"
  [hyperparameters]
  (let [params (merge default-hyperparameters hyperparameters)
        parameters (new svm_parameter)]
    (doseq [[param v] params]
      ;; let form here prevents linter from complaining about unused return val
      (let [x (Reflector/setInstanceField parameters (str/replace (name param) "-" "_") v)]
        x))
    parameters))

(defn train
  "Train a Linear SVM model for a given problem with specified parameters"
  [training-set-path hyperparameters]
  (let [problem-obj (training-set-path->problem training-set-path)
        params-obj (format-hyperparameters hyperparameters)]
    (when-let [error (svm/svm_check_parameter problem-obj params-obj)]
      (throw (Exception. error)))
    (svm/svm_train problem-obj params-obj)))

(defn save
  "Save a SVM model on the file system. Return the list of files that got saved
  on the file system."
  [model filepath]
  (svm/svm_save_model filepath model)
  [filepath])

(defn load
  "Load a SVM model from the file system into memory"
  [filepath]
  (svm/svm_load_model ^String filepath))

(defn- create-svm-node
  "Create a `svm_node` at `index` with `value`. If `value` is empty then it
  returns nil otherwise it returns the `svm_node`"
  [index value]
  (when-let [value (parsing/parse-double value)]
    (let [node (new svm_node)]
      (set! (. node index) (inc index))
      (set! (. node value) value)
      node)))

(defn predict
  "Predict the class/label of `features` given `model`. `features` is a vector
  of feature values. If the training set has been scaled before training, then
  `features` should be scaled with the same feature scaling function before
  being used to predict a class/label. The predicted class label is returned."
  [model _selected-features _hyperparameters feature-vector]
  (svm/svm_predict model (->> feature-vector
                              (keep-indexed create-svm-node)
                              into-array)))
