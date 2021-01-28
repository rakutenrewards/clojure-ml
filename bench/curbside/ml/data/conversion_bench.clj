(ns curbside.ml.data.conversion-bench
  (:require [libra.bench :refer :all]
            [libra.criterium :as c]
            [curbside.ml.data.conversion :refer :all]))

(defbench bench-feature-map-to-vector
  (let [feature-names [:a :b :c :d :e :f :g :h]
        feature-encoding {:features
                          {:b {:type :one-hot
                               :encoding-size 3
                               :one-hot-indices {"foo" 0, "bar" 1, "spam" 2}}}}
        feature-map {:a 1 :b "bar" :c 3 :d 3 :e nil :g 1.1 :h 56}]
    (is (c/bench
         (doall
          (feature-map-to-vector feature-names feature-encoding feature-map))))))
