(ns curbside.ml.utils.spec
  (:require
   [clojure.spec.alpha :as spec]
   [expound.alpha :as expound]))

(defn check
  "Returns true if provided data matches spec, throws an
  IllegalArgumentException otherwise."
  [spec data]
  (if (spec/valid? spec data)
    true
    (throw (IllegalArgumentException. (expound/expound-str spec data)))))
