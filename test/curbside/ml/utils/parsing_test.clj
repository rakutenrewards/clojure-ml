(ns curbside.ml.utils.parsing-test
  (:require
   [clojure.spec.alpha :as s]
   [clojure.spec.gen.alpha :as gen]
   [clojure.test :refer [deftest is testing]]
   [clojure.test.check.clojure-test :refer [defspec]]
   [clojure.test.check.properties :as prop]
   [curbside.ml.utils.parsing :as parsing]))

(deftest test-parse-or-identity
  (testing "parsing integer values"
    (is (= 1 (parsing/parse-or-identity "1")))
    (is (= -1 (parsing/parse-or-identity "-1")))
    (is (= 123456 (parsing/parse-or-identity "+123456"))))

  (testing "parsing floating point values"
    (is (== 1.123 (parsing/parse-or-identity "1.123")))
    (is (== -20.12 (parsing/parse-or-identity "-20.12")))
    (is (== -150.0 (parsing/parse-or-identity "-1.5e2")))
    (is (== 0.02 (parsing/parse-or-identity "2.0e-2")))
    (is (== 500.0 (parsing/parse-or-identity "5e2")))
    (is (== Double/POSITIVE_INFINITY (parsing/parse-or-identity "+Infinity")))
    (is (== Double/POSITIVE_INFINITY (parsing/parse-or-identity "Infinity")))
    (is (== Double/NEGATIVE_INFINITY (parsing/parse-or-identity "-Infinity")))
    (is (Double/isNaN (parsing/parse-or-identity "NaN"))))

  (testing "parsing booleans"
    (is (true? (parsing/parse-or-identity "true")))
    (is (false? (parsing/parse-or-identity "false"))))

  (testing "parsing string as-is"
    (letfn [(is-identity [s]
              (is (= s (parsing/parse-or-identity s))))]
      (is-identity "1/2")
      (is-identity "3.0f")
      (is-identity "this is a string")
      (is-identity "35 string starting with a number")
      (is-identity "2018-11-27T00:00:00.000Z"))))

(defspec parse-or-identity--generated-float
  200
  (prop/for-all [n (s/gen double?)]
    (try
      (let [parsed (parsing/parse-or-identity (str n))]
        (or (and (Double/isNaN n) (Double/isNaN parsed))
            (== n parsed)))
      (catch Throwable _t false))))

(defspec parse-or-identity--generated-int
  200
  (prop/for-all [n (s/gen int?)]
    (try
      (= n (parsing/parse-or-identity (str n)))
      (catch Throwable _t false))))

(deftest test-nan-to-nil
  (testing "Test the nan->nil macro"
    (is (nil? (parsing/nan->nil ((fn [] Double/NaN)))))
    (is (nil? (parsing/nan->nil ((fn [] (throw (Exception. "Exception.")))))))
    (is (= (parsing/nan->nil ((fn  [] "test"))) "test"))))
