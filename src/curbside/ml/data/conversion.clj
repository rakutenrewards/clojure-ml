(ns curbside.ml.data.conversion
  (:require
   [clojure.data.csv :as csv]
   [clojure.java.io :as io]
   [clojure.string :as string]
   [curbside.ml.data.encoding :as encoding]
   [curbside.ml.utils.parsing :as parsing])
  (:import
   (weka.core.converters CSVLoader CSVSaver)
   (weka.filters Filter)
   (weka.filters.unsupervised.attribute NumericToNominal)))

(defn csv-to-libsvm
  "convert a csv training set file into a libsvm one. the first column is the
  class, the other columns are the features.

   The format used by the LIBLINEAR library is the same as the LIBSVM one. The
   first value is the number that defines the class, usually [0 -1]. Then
   all features' value are separated by a space. Only the non-null features are
   written to the file. For instance, =1:0.5483= means that the first feature
   has value =0.5483=."
  [csv-file svm-file]
  (io/delete-file svm-file true)
  (with-open [reader (io/reader csv-file)
              writer (io/writer svm-file :append true)]
    (doseq [[class & features] (rest (csv/read-csv reader))]
      (.write writer (str class " " (->> features
                                         (map-indexed (fn [feature value]
                                                        (if (empty? value)
                                                          ""
                                                          (str (inc feature) ":" value " "))))
                                         (apply str)
                                         clojure.string/trim) "\n")))))

(defn csv-to-arff
  "convert a csv training set file into a ARFF one. Returns the ARFF object in
  memory.

  The [ARFF (Attribute-Relation File Format)](https://www.cs.waikato.ac.nz/ml/weka/arff.html)
  is Weka's internal training set format. It is basically a CSV file with
  special header information that describes each of the columns.

  With the ARFF format, when we take a CSV file and convert it into ARFF. The
  conversion handles both numerical and categorical columns. Missing values are
  also properly marked as `?` values in the resulting ARFF. Depending on the
  algorithm used the type of the =label= attribute needs to be =nominal= or
  =numeric=. This is the reason why we have a =class-type= optional parameter
  that will instruct the function how we have to define the =label= attribute."
  ([csv-file predictor-type]
   (let [csv-file (if (string? csv-file) (io/as-file csv-file) csv-file)
         loader (doto (CSVLoader.) (.setSource csv-file))
         instances (doto (.getDataSet loader) (.setClassIndex 0))]
     (if (= :classification predictor-type)
       (let [filter (doto (NumericToNominal.)
                      (.setOptions (into-array String ["-R" "first"]))
                      (.setInputFormat instances))]
         (Filter/useFilter instances filter))
       instances)))
  ([csv-file arff-file predictor-type]
   (let [arff (csv-to-arff csv-file predictor-type)]
     (with-open [writer (io/writer arff-file)]
       (.write writer (.toString arff)))
     arff)))

(defn arff-to-csv
  [dataset sampled-dataset-file]
  (let [sampled-dataset-arff-file (string/replace sampled-dataset-file ".csv" ".arff")
        csv-saver (CSVSaver.)]
    (with-open [writer (io/writer sampled-dataset-arff-file)]
      (.write writer (.toString dataset)))
    (CSVSaver/runFileSaver csv-saver (into-array String ["-i" sampled-dataset-arff-file
                                                         "-o" sampled-dataset-file]))
    ;; Removing the `?` character for missing values
    ;; This can't be done with the `CSVSaver` API since
    ;; it doesn't accept empty values...
    (spit sampled-dataset-file (-> (slurp sampled-dataset-file)
                                   (string/replace ",?," ",,")
                                   (string/replace ",?" ",")))))

(defn csv-column-keys
  "Returns the keys in the CSV's header. The keys are put in a vector in the same
  order they appear in the CSV "
  [csv-path]
  (with-open [reader (io/reader csv-path)]
    (mapv keyword (first (csv/read-csv reader)))))

(defn- parse-row-values
  [csv-row]
  (mapv parsing/parse-or-identity csv-row))

(defn csv-to-maps
  "Converts a csv training set to a vector of maps, where each map has the same
  keys, as defined in the header of the CSV file.

  Where converting a training set to CSV, the order of the column is important,
  as machine learning algorithms training directly on CSV files will use the
  order of the columns as the order of the features. Therefore, When converting
  from maps to a CSV file, a vector of keys in order must be supplied."
  [csv-path]
  (with-open [reader (io/reader csv-path)]
    (let [data (csv/read-csv reader)
          header (map keyword (first data))
          rows (map parse-row-values (rest data))]
      (mapv zipmap
            (repeat header)
            rows))))

(defn- valid-keys-for-header?
  [keys-in-order maps]
  (or (empty? maps)
      (= (set keys-in-order)
         (set (keys (first maps))))))

(defn- ratios->doubles
  [row]
  (map (fn [n]
         (if (ratio? n)
           (double n)
           n))
       row))

(defn maps-to-csv
  "Writes a sequence of `maps` (which are assumed to contain the same keys) to a
  CSV at `output-path`. The columns are in the order specified by
  `keys-in-order`. `keys-in-order` must contains all the keys present in the
  maps."
  [output-path keys-in-order maps]
  {:pre [(valid-keys-for-header? keys-in-order maps)]}
  (let [header (map name keys-in-order)
        rows (if (empty? keys-in-order)
               []
               (->> maps
                    (map (apply juxt keys-in-order) maps)
                    (map ratios->doubles)))]
    (with-open [writer (io/writer output-path)]
      (csv/write-csv writer (concat [header] rows)))))

(defn vector-to-csv
  [output-path header v]
  (with-open [writer (io/writer output-path)]
    (csv/write-csv writer (concat [[header]]
                                  (map vector v)))))

(defn feature-map-to-vector
  "Converts a map of features to a vector using a vector of feature-names."
  ([feature-names feature-map]
   (->> feature-names
        (map (fn [n] (get feature-map n)))
        (flatten)
        (into [])))
  ([feature-names dataset-encoding feature-map]
   (if (some? dataset-encoding)
     (->> feature-map
          (encoding/encode-feature-map dataset-encoding)
          (feature-map-to-vector feature-names))
     (feature-map-to-vector feature-names feature-map))))
