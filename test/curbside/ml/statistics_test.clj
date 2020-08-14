(ns curbside.ml.statistics-test
  (:require
   [clojure.test :refer [deftest is testing use-fixtures]]
   [curbside.ml.statistics :as stats]))

(deftest test-descriptive-stats
  (let [xs [7 7 31 31 47 75 87 115 116 119 119 155 177]]
    (is (= {:mean (double (/ 1086 13))
            :median 87.0
            :q1 31.0
            :q3 119.0
            :iqr 88.0}
           (stats/descriptive-stats xs)))))

(def some-numbers-with-outliers [1 2 10 2 33 5 0 4 -100])
(def some-maps-with-outliers [{:a 0   :b 10}
                              {:a 1   :b -1}
                              {:a 1   :b -1}
                              {:a 2   :b 0}
                              {:a 2   :b 0}
                              {:a 3   :b 1}
                              {:a 100 :b 1}])

(deftest test-iqr-outliers-mask
  (is (= [false false false false true false false false true]
         (stats/iqr-outliers-mask some-numbers-with-outliers))))

(deftest test-mask-logical-or
  (is (= [true false true true true]
       (#'stats/masks-logical-or [[true false false true  true]
                                  [true false false true  true]
                                  [true false true  false true]]))))

(deftest test-remove-iqr-outliers
  (is (= [1 2 10 2 5 0 4]
         (stats/remove-iqr-outliers some-numbers-with-outliers)))
  (is (= [{:a 1 :b -1}
          {:a 1 :b -1}
          {:a 2 :b 0}
          {:a 2 :b 0}
          {:a 3 :b 1}]
         (stats/remove-iqr-outliers [:a :b] some-maps-with-outliers))))
