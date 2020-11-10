(ns curbside.ml.utils.io
  (:import
   (java.io File)))

(defn create-temp-path
  [extension]
  (let [file (doto (File/createTempFile "temp_" extension)
               (.deleteOnExit))]
    (.getPath file)))

(defn create-temp-csv-path
  ([]
   (create-temp-path ".csv"))
  ([content]
   (let [path (create-temp-csv-path)]
     (spit path content)
     path)))
