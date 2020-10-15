(ns curbside.ml.metrics-test
  (:require [curbside.ml.metrics :as metrics]
            [clojure.test :refer [deftest is testing]]))

(def a-dataset
  {:feature-maps [{:a 0, :b 0}
                  {:a 2, :b -1}
                  {:a 2, :b -1}
                  {:a 2, :b -1}
                  {:a 2, :b -1}]
   :features [:a :b]
   :labels [0.0 1.0 1.0 0.0 0.0]})

(deftest feature-metrics
  (is (= {:cfs-subset [:a],
          :correlation {:b 0.408248290463863, :a 0.408248290463863},
          :relief-f {:b -0.06666666666666668, :a -0.06666666666666668}}
         (metrics/feature-metrics
          a-dataset
          :classification
          [:cfs-subset :correlation :relief-f]))))
