(ns curbside.ml.utils.date-time-test
  (:require
   [clj-time.types :as time-types]
   [clojure.test :refer [deftest is testing]]
   [curbside.ml.utils.date-time :as date-time]))

(deftest parse
  (testing "given string dates in ISO 8601 formats, when parsing them, then a date-time object is returned."
    (is (time-types/date-time? (date-time/parse "2019-01-01T12:00:00")))
    (is (time-types/date-time? (date-time/parse "2019-01-01T12:00:00.000Z")))
    (is (time-types/date-time? (date-time/parse "2019-01-01T12:00:00.000-04")))
    (is (time-types/date-time? (date-time/parse "2019-01-01T12:00:00Z"))))

  (testing "given string dates in SQL friendly format, when parsing them, then a date-time object is returned."
    (is (time-types/date-time? (date-time/parse "2019-01-01 12:00:00")))
    (is (time-types/date-time? (date-time/parse "2019-01-01 12:00:00.000Z")))
    (is (time-types/date-time? (date-time/parse "2019-01-01 12:00:00.000-04")))
    (is (time-types/date-time? (date-time/parse "2019-01-01 12:00:00Z")))))
