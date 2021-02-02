(ns curbside.ml.utils.spec
  (:require
   [clojure.spec.alpha :as s]
   [expound.alpha :as expound]))

(s/def ::finite-double
  (s/double-in :NaN? false :infinite? false))

(s/def ::finite-number
  (s/with-gen
    (s/and
     (s/conformer
      (fn [x]
        (try
          (double x)
          (catch Exception _e
            ::s/invalid))))
     ::finite-double)
    #(s/gen ::finite-double)))

(defn check
  "Returns true if provided data matches spec, throws an
  IllegalArgumentException otherwise."
  [spec data]
  (if (s/valid? spec data)
    true
    (throw (IllegalArgumentException. (expound/expound-str spec data)))))
