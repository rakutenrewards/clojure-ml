(ns curbside.ml.data.features.stats-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.data.features.stats :as stats]))

(deftest num-distinct-values
  (is (= 0 (stats/num-distinct-values [])))
  (is (= 1 (stats/num-distinct-values [1])))
  (is (= 1 (stats/num-distinct-values [1 1])))
  (is (= 2 (stats/num-distinct-values [1 1.0])))
  (is (= 3 (stats/num-distinct-values [1 1.0 "foo"]))))

(deftest num-integer-values
  (is (= 0 (stats/num-integer-values [])))
  (is (= 1 (stats/num-integer-values [1])))
  (is (= 2 (stats/num-integer-values [1 1])))
  (is (= 1 (stats/num-integer-values [1 1.0])))
  (is (= 1 (stats/num-integer-values [1 nil 1.0 "foo"]))))

(deftest num-missing-values
  (is (= 0 (stats/num-missing-values [])))
  (is (= 0 (stats/num-missing-values [1 2 3])))
  (is (= 1 (stats/num-missing-values [1 2 3 nil])))
  (is (= 2 (stats/num-missing-values [1 nil 2 3 nil]))))

(deftest num-real-values
  (is (= 0 (stats/num-real-values [])))
  (is (= 1 (stats/num-real-values [1.0])))
  (is (= 1 (stats/num-real-values [1.0 2 3])))
  (is (= 2 (stats/num-real-values [(double 1.0) (float 2.0)]))))

(deftest num-total-values
  (is (= 0 (stats/num-total-values [])))
  (is (= 1 (stats/num-total-values [nil])))
  (is (= 4 (stats/num-total-values [1 2 3 "toto"]))))

(deftest num-unique-values
  (is (= 0 (stats/num-unique-values [])))
  (is (= 1 (stats/num-unique-values [nil])))
  (is (= 0 (stats/num-unique-values [1 1])))
  (is (= 2 (stats/num-unique-values [1 1.0])))
  (is (= 2 (stats/num-unique-values [1 2 2 "three" 4 4]))))

(deftest numeric-stats
  (is (= {:max 6
          :mean 3.5
          :min 1
          :standard-deviation 1.8708286933869707
          :sum 21.0
          :sum-squared 91.0}
         (#'stats/numeric-stats [1 2 3 4 5 6]))))

(def a-datasets
  {:features [:a :b]
   :feature-maps [{:a 1, :b -10}
                  {:a "test", :b -20}
                  {:a 3, :b -20}
                  {:a nil, :b 40}]
   :labels [1 2 3 4]})

(def feature-statistics-of-:a
  {:num-distinct-values 4
   :num-integer-values 2
   :num-missing-values 1
   :num-real-values 0
   :num-total-values 4
   :num-unique-values 4})

(def feature-statistics-of-:b
  {:min -20
   :num-unique-values 2
   :mean -2.5
   :num-distinct-values 3
   :sum-squared 2500.0
   :num-real-values 0
   :standard-deviation 28.722813232690143
   :num-integer-values 4
   :max 40
   :num-missing-values 0
   :sum -10.0
   :num-total-values 4})

(deftest feature-statistics
  (testing "feature-statistics of a feature with only numeric values"
    (is (= feature-statistics-of-:b
           (stats/feature-statistics a-datasets :b))))
  (testing "feature-statistics of a feature with non-numeric values"
    (is (= feature-statistics-of-:a
           (stats/feature-statistics a-datasets :a)))))

(deftest dataset-statistics
  (is (= {:a feature-statistics-of-:a
          :b feature-statistics-of-:b}
         (stats/dataset-statistics a-datasets))))
