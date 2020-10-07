(ns curbside.ml.utils.weka
  (:require
   [clojure.java.io :as io])
  (:import
   (java.util ArrayList)
   (weka.core Attribute DenseInstance Instances)))

(defn problem
  "Define a problem space by reading an ARFF training set. If training is an ARFF
  file then the problem will be read from that file. If training is a set of Instances
  then that set of Instances will be returned. `class-col-index` is the index of the
  column where the class is represented."
  [dataset & {:keys [class-col-index]}]
  (let [instances (if (and (string? dataset) (.exists (io/as-file dataset)))
                    (with-open [reader (io/reader dataset)]
                      (Instances. reader))
                    dataset)]
    (.setClassIndex instances (or class-col-index 0))
    instances))

(defn- attribute-list
  [predictor-type selected-features]
  (->> selected-features
       (map name)
       (map #(Attribute. %))
       (cons (if (= :classification predictor-type)
               (Attribute. "@@class@@" ["0.0" "1.0"])
               (Attribute. "@@class@@")))
       (#(ArrayList. %))))

(defn create-instance
  [predictor-type selected-features feature-vector]
  (let [attributes (attribute-list predictor-type selected-features)
        dataset (doto (Instances. "test-instances" attributes 1)
                  (.setClassIndex 0))
        instance (DenseInstance. (.numAttributes dataset))]
    (doseq [[attribute val] (map vector (rest attributes) feature-vector)]
      (when val
        (.setValue instance attribute (double val))))
    (.setDataset instance dataset)
    instance))
