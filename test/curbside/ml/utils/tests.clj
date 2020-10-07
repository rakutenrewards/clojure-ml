(ns curbside.ml.utils.tests
  (:require
   [clojure.data.csv :as csv]
   [clojure.edn :as edn]
   [clojure.java.io :as io]
   [clojure.walk :as walk]
   [conjure.core :as conjure]))

(defn count-csv-rows
  [path]
  (with-open [reader (io/reader path)]
    (count (rest (csv/read-csv reader)))))

(defn resource-name-to-path-str
  [r]
  (.getPath (io/resource r)))

(def dummy-regression-single-label-dataset-path
  (io/resource "datasets/dummy-regression-single-label.csv"))

(def dummy-example-weights-path (io/resource "datasets/dummy-weights.csv"))

(def dummy-ranking-dataset-path
  (io/resource "datasets/dummy-ranking.csv"))

(def dummy-ranking-dataset-groups-path
  (io/resource "datasets/dummy-ranking.groups.csv"))

(def dummy-ranking-dataset-encoding
  (edn/read-string (slurp (io/resource "datasets/dummy-ranking.encoding.edn"))))

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
