(ns curbside.ml.training-sets.sampling-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.training-sets.sampling :refer [sample-training-set]]
   [curbside.ml.utils.io :as io-utils]
   [curbside.ml.utils.tests :as tutils]))

(def an-empty-training-set
  {:feature-maps []
   :features [:a :b]
   :labels []})

(def a-training-set
  {:feature-maps (vec (repeat 1000 {:a 0 :b 1}))
   :features [:a :b]
   :labels (vec (range 1000))
   :weights (vec (repeat 1000 1.0))})

(deftest test-sample-training-set
  (testing "given the default config, when sampling, then it keeps all the data points"
    (is (= 1000
           (count (:feature-maps (sample-training-set a-training-set {}))))))

  (testing "given given a :max-sample-size config, when sampling, the right amount of points is sampled"
    (is (= 200
           (count (:feature-maps (sample-training-set a-training-set {:max-sample-size 200}))))))

  (testing "given given a :sample-size-percent config, when sampling, the right amount of points is sampled"
    (is (= 250
           (count (:feature-maps (sample-training-set a-training-set {:sample-size-percent 25}))))))

  (testing "given an empty dataset and a :max-sample-size config, when sampling, an empty dataset is produced"
    (is (= an-empty-training-set
           (sample-training-set an-empty-training-set {:max-sample-size 1000}))))

  (testing "given an empty dataset and a :sample-size-percent config, when sampling, an empty dataset is produced"
    (is (= an-empty-training-set
           (sample-training-set an-empty-training-set {:sample-size-percent 55})))))
