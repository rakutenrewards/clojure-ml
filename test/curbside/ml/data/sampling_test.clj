(ns curbside.ml.data.sampling-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.data.sampling :refer [sample-dataset]]
   [curbside.ml.utils.io :as io-utils]
   [curbside.ml.utils.tests :as tutils]))

(def an-empty-dataset
  {:feature-maps []
   :features [:a :b]
   :labels []})

(def a-dataset
  {:feature-maps (vec (repeat 1000 {:a 0 :b 1}))
   :features [:a :b]
   :labels (vec (range 1000))
   :weights (vec (repeat 1000 1.0))})

(deftest test-sample-dataset
  (testing "given the default config, when sampling, then it keeps all the data points"
    (is (= 1000
           (count (:feature-maps (sample-dataset a-dataset {}))))))

  (testing "given given a :max-sample-size config, when sampling, the right amount of points is sampled"
    (is (= 200
           (count (:feature-maps (sample-dataset a-dataset {:max-sample-size 200}))))))

  (testing "given given a :sample-size-percent config, when sampling, the right amount of points is sampled"
    (is (= 250
           (count (:feature-maps (sample-dataset a-dataset {:sample-size-percent 25}))))))

  (testing "given an empty dataset and a :max-sample-size config, when sampling, an empty dataset is produced"
    (is (= an-empty-dataset
           (sample-dataset an-empty-dataset {:max-sample-size 1000}))))

  (testing "given an empty dataset and a :sample-size-percent config, when sampling, an empty dataset is produced"
    (is (= an-empty-dataset
           (sample-dataset an-empty-dataset {:sample-size-percent 55})))))
