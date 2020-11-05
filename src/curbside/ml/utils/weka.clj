(ns curbside.ml.utils.weka
  (:require
   [clojure.java.io :as io]
   [curbside.ml.data.conversion :as conversion])
  (:import
   (java.io PipedInputStream PipedOutputStream)
   (java.util ArrayList)
   (weka.core Attribute DenseInstance Instances)))

(defn dataset->weka-instances
  "Convert a dataset to the the Weka Instances class."
  [{:keys [features feature-maps labels] :as _dataset} predictor-type]
  (with-open [csv-output-stream (PipedOutputStream.)
              csv-input-stream (PipedInputStream. csv-output-stream)]
    (future ;; when using piped streams, reading and writing must be in separate thread or else we can deadlock.

      ;; A new output stream is created for this specific thread. This ensures
      ;; that the stream is closed in case of an exception, or else this would
      ;; cause a deadlock. This happens when thread writing to the output stream
      ;; dies while the thread reading the input stream waits.
      (with-open [csv-output-stream (io/output-stream csv-output-stream)]
        (conversion/maps-to-csv csv-output-stream
                                (cons :label features)
                                (map #(assoc %1 :label %2) feature-maps labels))))
    (conversion/csv-to-arff csv-input-stream predictor-type)))

(defn- attribute-list
  [predictor-type selected-features]
  (->> selected-features
       (map name)
       (map #(Attribute. %))
       (cons (if (= :classification predictor-type)
               (Attribute. "label" ["0.0" "1.0"])
               (Attribute. "label")))
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
