(ns curbside.ml.data.scaling-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.data.scaling :as scaling]
   [curbside.ml.utils.tests :as tutils]))

(def a-feature-map {:x 10 :y  -2 :z 1000 :unknown nil})

(def min-max-features-factors {:x {:min 0 :max 20} :y {:min -5 :max 5}})

(deftest test-min-max-scaling
  (testing "given a number, when scaling and unscaling it, then it still have the same value"
    (is (as-> 2 value
          (scaling/apply-scaling :min-max value {:min 0 :max 10})
          (scaling/apply-unscaling :min-max value {:min 0 :max 10})
          (== 2 value))))

  (testing "given a number, when applying scaling, then the number is scaled"
    (is (== 0.2 (scaling/apply-scaling :min-max 2 {:min 0 :max 10}))))

  (testing "given a number, when applying unscaling, then the number is unscaled"
    (is (== -10 (scaling/apply-unscaling :min-max 0 {:min -10 :max 0}))))

  (testing "given a feature map and scaling factors, when applying scaling, all features in the factor map are scaled"
    (let [scaled-map (scaling/scale-map-keys :min-max a-feature-map min-max-features-factors)]
      (is (== 0.5 (:x  scaled-map)))
      (is (== 0.3 (:y  scaled-map)))
      (is (== 1000 (:z  scaled-map)))
      (is (nil? (:unknown  scaled-map))))))

(def log10-features-factors {:x {} :z {}})

(deftest test-log10-scaling
  (testing "given a number, when scaling and unscaling it, then it still have the same value"
    (is (as-> 2 value
          (scaling/apply-scaling :log10 value {})
          (scaling/apply-unscaling :log10 value {})
          (== 2 value))))

  (testing "given a number, when applying scaling, then the number is scaled"
    (is (== 1 (scaling/apply-scaling :log10 10 nil))))

  (testing "given a number, when applying unscaling, then the number is unscaled"
    (is (== 10 (scaling/apply-unscaling :log10 1 nil))))

  (testing "given a feature map and scaling factors, when applying scaling, all features in the factor map are scaled"
    (let [scaled-map (scaling/scale-map-keys :log10 a-feature-map log10-features-factors)]
      (is (== 1 (:x  scaled-map)))
      (is (== -2 (:y  scaled-map)))
      (is (== 3 (:z  scaled-map)))
      (is (nil? (:unknown  scaled-map))))))

(deftest test-log10-scaling-min-max-values
  (testing "given a negative value, when applying scaling, it returns a small value instead of negative infinity"
    (is (== scaling/min-log10-value (scaling/apply-scaling :log10 -2 nil)))))

(def a-dataset
  {:feature-maps [{:a 0, :b 0}
                  {:a 2, :b -1}
                  {:a 6, :b -0.5}
                  {:a 10, :b 0}]
   :features [:a :b]
   :labels [10 100 1000 10000]})

(def expected-factors {:features [{:a {:min 0 :max 10}
                                   :b {:min -1 :max 0}}]
                       :labels [{}]})

(def expected-scaled-dataset
  {:feature-maps [{:a 0.0, :b 1.0}
                  {:a 0.2, :b 0.0}
                  {:a 0.6, :b 0.5}
                  {:a 1.0, :b 1.0}]
   :features [:a :b]
   :labels [1.0 2.0 3.0 4.0]})

(deftest test-scale-dataset
  (testing "given a training set, when scaling, all features and labels are scaled"
    (let [[scaled-dataset factors] (scaling/scale-dataset [:min-max]
                                                          [:log10]
                                                          a-dataset)]
      (is (tutils/approx= expected-scaled-dataset scaled-dataset 1e-6))
      (is (= expected-factors factors)))))

(def dataset-scaling-factors {:features [min-max-features-factors]
                              :labels [{}]})

(deftest test-unscale-label
  (testing "given an inferred value, when unscaling, the value is unscaled"
    (is (== 0.1 (scaling/unscale-label [:log10] dataset-scaling-factors -1))))
  (testing "given multiple scaling functions, when unscaling, the value is unscaled."
    (let [factors {:features [] :labels [{:min 0 :max 100} {}]}]
      (is (== 100 (scaling/unscale-label [:min-max :log10] factors 0))))))
