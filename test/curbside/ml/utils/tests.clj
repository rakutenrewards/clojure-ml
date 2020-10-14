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

(declare approx=)

(defn- approx=-seqs
  [x y epsilon]
  (and
   (= (count x) (count y))
   (every? true? (map #(approx= %1 %2 epsilon) x y))))

(defn- approx=-maps
  [x y epsilon]
  (let [x-ks (keys x)]
    (and
     (= (set x-ks) (set (keys y)))
     (every? true? (map #(approx= %1 %2 epsilon)
                        (vals x)
                        (map #(get y %) x-ks)))))) ;; ensure the same key order for x and y

(defn approx=
  [x y epsilon]
  (cond
    (and (number? x)
         (number? y))
    (< (Math/abs (- x y)) epsilon)

    (and (keyword? x)
         (keyword? y))
    (= x y)

    (and (sequential? x)
         (sequential? y))
    (approx=-seqs x y epsilon)

    (and (map? x)
         (map? y))
    (approx=-maps x y epsilon)

    (= (type x) (type y)) ;; any type but the ones above
    (= x y)

    :else
    false))

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
