(ns curbside.ml.utils.date-time
  (:require
   [clj-time.format :as time-format])
  (:import
   (org.joda.time DateTimeZone)))

(def iso-date-time-formatter
  (time-format/formatter DateTimeZone/UTC
                         ;; ISO 8601 formats
                         "yyyy-MM-dd'T'HH:mm:ss.SSSSSSZZ"
                         "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
                         "yyyy-MM-dd'T'HH:mm:ss.SSSZZ"
                         "yyyy-MM-dd'T'HH:mm:ss.SSS"
                         "yyyy-MM-dd'T'HH:mm:ssZZ"
                         "yyyy-MM-dd'T'HH:mm:ss"

                         ;; Postgres friendly formats
                         "YYYY-MM-dd HH:mm:ss.SSSSSSZZ"
                         "YYYY-MM-dd HH:mm:ss.SSSSSS"
                         "YYYY-MM-dd HH:mm:ss.SSSZZ"
                         "YYYY-MM-dd HH:mm:ss.SSS"
                         "YYYY-MM-dd HH:mm:ssZZ"
                         "YYYY-MM-dd HH:mm:ss"))

(defn parse
  "Given a date-time string, returns a DateTime object. First attempts
   various ISO-8601 and Postgres formats."
  [date-time]
  (when (not-empty date-time)
    (time-format/parse iso-date-time-formatter date-time)))
