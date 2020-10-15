(ns curbside.ml.utils.weka
  (:require
   [clojure.java.io :as io]
   [curbside.ml.data.conversion :as conversion])
  (:import
   (java.util ArrayList)
   (weka.core Attribute DenseInstance Instances)))

(defn- attribute-list
  [predictor-type selected-features]
  (->> selected-features
       (map name)
       (map #(Attribute. %))
       (cons (if (= :classification predictor-type)
               (Attribute. "label" ["0.0" "1.0"])
               (Attribute. "label")))
       (#(ArrayList. %))))

(defn dataset->weka-instances
  "Convert a dataset to the the Weka Instances class."
  [{:keys [features feature-maps labels weights] :as _dataset} predictor-type]
  (let [weights (or weights (repeat 1.0))
        instances (Instances. "" (attribute-list predictor-type features) (count labels))]
    (doseq [[feature-map label weight] (map vector feature-maps labels weights)]
      (let [feature-vector (conversion/feature-map-to-vector features feature-map)]
        (.add instances (DenseInstance. (double weight) (double-array (cons label feature-vector))))))
    (.setClassIndex instances 0)
    instances))

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
