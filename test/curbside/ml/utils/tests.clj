(ns curbside.ml.utils.tests
  (:require
   [clojure.data.csv :as csv]
   [clojure.edn :as edn]
   [clojure.walk :as walk]
   [conjure.core :as conjure]
   [clojure.java.io :as io])
  (:import
   [java.io File]))

(defn create-temp-path
  [extension]
  (let [file (doto (File/createTempFile "test_" extension)
               (.deleteOnExit))]
    (.getPath file)))

(defn create-temp-csv-path
  ([]
   (create-temp-path ".csv"))
  ([content]
   (let [path (create-temp-csv-path)]
     (spit path content)
     path)))

(defn count-csv-rows
  [path]
  (with-open [reader (io/reader path)]
    (count (rest (csv/read-csv reader)))))

(defn resource-name-to-path-str
  [r]
  (.getPath (io/resource r)))

(def dummy-regression-single-label-training-set-path
  (io/resource "training-sets/dummy-regression-single-label.csv"))

(def dummy-example-weights-path (io/resource "training-sets/dummy-weights.csv"))

(def dummy-ranking-training-set-path
  (io/resource "training-sets/dummy-ranking.csv"))

(def dummy-ranking-training-set-groups-path
  (io/resource "training-sets/dummy-ranking.groups.csv"))

(def dummy-ranking-training-set-encoding
  (edn/read-string (slurp (io/resource "training-sets/dummy-ranking.encoding.edn"))))

(defn approx=
  [x y tolerance]
  (< (Math/abs (- x y)) tolerance))

(defmacro stubbing-private
  "Allows using Conjure's `stubbing` macro with private definitions."
  [bindings & body]
  (let [kvs      (partition 2 bindings)
        privates (map first  kvs)
        stubbing (map second kvs)
        defnames (map (comp gensym name) privates)]
    `(do ~@(map (fn [x] `(defn ~x [& _#])) defnames)
         (conjure/stubbing ~(vec (interleave defnames stubbing))
           (with-redefs ~(vec (interleave privates defnames))
             ~@(walk/postwalk-replace (zipmap privates defnames) body))))))
