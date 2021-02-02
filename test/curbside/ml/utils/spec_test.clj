(ns curbside.ml.utils.spec-test
  (:require
   [clojure.spec.alpha :as s]
   [clojure.spec.gen.alpha :as gen]
   [clojure.test :refer [deftest is]]
   [curbside.ml.utils.spec :as spec]))

(deftest finite-number-spec
  (is (s/valid? ::spec/finite-number (int 1)))
  (is (s/valid? ::spec/finite-number (bigint 1)))
  (is (s/valid? ::spec/finite-number (double 1.0)))
  (is (s/valid? ::spec/finite-number (float 1.0)))

  (is (not (s/valid? ::spec/finite-number ##NaN)))
  (is (not (s/valid? ::spec/finite-number "adsf")))
  (is (not (s/valid? ::spec/finite-number :asdf)))
  (is (not (s/valid? ::spec/finite-number Double/POSITIVE_INFINITY)))
  (is (not (s/valid? ::spec/finite-number Double/NEGATIVE_INFINITY)))
  (is (not (s/valid? ::spec/finite-number Float/POSITIVE_INFINITY)))

  (is (s/valid? ::spec/finite-number
                (gen/generate (s/gen ::spec/finite-number)))
      "ensure that we can use this spec for generative testing"))
