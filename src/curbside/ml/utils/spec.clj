(ns curbside.ml.utils.spec
  (:require
   [clojure.spec.alpha :as s]
   [expound.alpha :as expound]))

(s/def ::finite-number
  (s/and
   (s/or :int int?
         :finite-double (s/double-in :NaN? false :infinite? false))
   (s/conformer second))) ;; Allows to compose ::finite-number with other predicates such as pos? without having to unform the tuple e.g. `[:int 10]`

(defn check
  "Returns true if provided data matches spec, throws an
  IllegalArgumentException otherwise."
  [spec data]
  (if (s/valid? spec data)
    true
    (throw (IllegalArgumentException. (expound/expound-str spec data)))))
