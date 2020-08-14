(ns curbside.ml.training-sets.conversion-test
  (:require
   [clojure.string :as string]
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.training-sets.conversion :as conversion]
   [curbside.ml.utils.tests :as tutils]))

(def an-empty-csv "label,a,b,c\n")
(def a-header-only-csv "label,a,b,c\n")

(def a-csv
  (string/join "\n"
               ["label,a,b,c"
                "1.0,2.0,3.0,2.0"
                "10,1,0,0\n"]))
(def some-maps [{:b 3.0
                 :a 2.0
                 :label 1.0
                 :c 2.0}
                {:a 1
                 :b 0
                 :c 0
                 :label 10}])

(def a-csv-with-missing-values
  (string/join "\n"
               ["label,a,b,c"
                "23.0,,2.0,"
                ",1.0,,0.0\n"]))
(def some-maps-with-nil-keys [{:label 23.0
                               :a nil
                               :b 2.0
                               :c nil}
                              {:label nil
                               :a 1.0
                               :b nil
                               :c 0.0}])

(def a-csv-with-boolean-label
  (string/join "\n"
               ["label,a"
                "true,2.0"
                "false,10.0"]))

(def a-csv-with-string-labels
  (string/join "\n"
               ["label,a"
                "cat,2.0"
                "dog,10.0\n"]))
(def some-maps-with-string-labels [{:label "cat"
                                    :a 2.0}
                                   {:label "dog"
                                    :a 10.0}])

(def some-maps-with-ratios
  [{:a (/ 1 4)
    :label (/ 1 2)}])

(def a-csv-with-ratios-converted-to-doubles
  "label,a\n0.5,0.25\n")

(defn is-csv-to-maps-conversion-valid?
  [csv-content expected-maps]
  (let [csv-path (tutils/create-temp-csv-path)]
    (spit csv-path csv-content)
    (is (= expected-maps (conversion/csv-to-maps csv-path)))))

(deftest test-csv-to-maps
  (testing "testing conversion from csv to maps"
    (is-csv-to-maps-conversion-valid? an-empty-csv [])
    (is-csv-to-maps-conversion-valid? a-header-only-csv [])
    (is-csv-to-maps-conversion-valid? a-csv some-maps)
    (is-csv-to-maps-conversion-valid? a-csv-with-missing-values some-maps-with-nil-keys)
    (is-csv-to-maps-conversion-valid? a-csv-with-boolean-label
                                      [{:label true
                                        :a 2.0}
                                       {:label false
                                        :a 10.0}])
    (is-csv-to-maps-conversion-valid? a-csv-with-string-labels some-maps-with-string-labels)))

(defn is-maps-to-csv-conversion-valid?
  [expected-csv-content maps column-keys]
  (let [csv-path (tutils/create-temp-csv-path)]
    (conversion/maps-to-csv csv-path column-keys maps)
    (is (= expected-csv-content (slurp csv-path)))))

(deftest test-maps-to-csv
  (testing "testing conversion from maps to csv"
    (is-maps-to-csv-conversion-valid? a-header-only-csv
                                      []
                                      [:label :a :b :c])
    (is-maps-to-csv-conversion-valid? a-csv
                                      some-maps
                                      [:label :a :b :c])
    (is-maps-to-csv-conversion-valid? a-csv-with-missing-values
                                      some-maps-with-nil-keys
                                      [:label :a :b :c])
    (is-maps-to-csv-conversion-valid? a-csv-with-string-labels
                                      some-maps-with-string-labels
                                      [:label :a])
    (is-maps-to-csv-conversion-valid? a-csv-with-ratios-converted-to-doubles
                                      some-maps-with-ratios
                                      [:label :a])))

(deftest test-feature-map-to-vector
  (testing "given a feature map, when converting to vector, only selected features are kept"
    (is (= [1 2 3] (conversion/feature-map-to-vector [:a :b :c] {:a 1 :b 2 :c 3 :d "danger"}))))
  (testing "given a feature map, when converting to vector, features are put in the order of the inputed selected features"
    (is (= [1 2 3 4] (conversion/feature-map-to-vector [:b-2 :c :b-1 :a]
                                                       {:c 2 :b-2 1 :a 4 :b-1 3})))))
