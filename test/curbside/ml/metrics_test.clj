(ns curbside.ml.metrics-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.metrics :as metrics]))

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


(deftest optimization-type
  (is (= :minimize (metrics/optimization-type :mean-absolute-error)))
  (is (= :maximize (metrics/optimization-type :ndcg)))
  (is (thrown? Exception (metrics/optimization-type :unknow-metric-setting-a-default-would-be-dangerous))))
