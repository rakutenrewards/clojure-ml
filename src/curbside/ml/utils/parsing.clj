(ns curbside.ml.utils.parsing
  (:require
   [clojure.string :as str]))

(defn parse-double
  "Parse a string into a double. If the string is empty return nil. If it is a
  double, return it. If it is an integer, type cast it into a double and return
  it."
  [s]
  (cond
    (and (string? s) (empty? s))
    nil

    (float? s)
    s

    (integer? s)
    (double s)

    (and (string? s) (= "true" (str/lower-case s)))
    1.0

    (and (string? s) (= "false" (str/lower-case s)))
    0.0

    (string? s)
    (Double/parseDouble s)))

(defn parse-float
  "Parse a string into a float. If the string is empty return nil. If it is a
  double, return it. If it is an integer, type cast it into a float and return
  it."
  [s]
  (cond
    (and (string? s) (empty? s))
    nil

    (float? s)
    s

    (integer? s)
    (float s)

    (and (string? s) (= "true" (str/lower-case s)))
    (float 1.0)

    (and (string? s) (= "false" (str/lower-case s)))
    (float 0.0)

    (string? s)
    (Float/parseFloat s)))

(def int-regex #"[-+]?[\d]+")
(def double-regex #"[-+]?[\d]+\.?[\d]*(?:[eE][-+]?[\d]+)?")

(defn parse-or-identity
  "Tries to parse either a long, a double or a boolean from `s`. If `s` does not contain a
  number, returns `s`."
  [s]
  (cond
    (empty? s)
    nil

    (re-matches int-regex s)
    (Long/parseLong s)

    (re-matches double-regex s)
    (Double/parseDouble s)

    (or (= "Infinity" s) (= "+Infinity" s)) ;; Regex not used here for speed
    Double/POSITIVE_INFINITY

    (= "-Infinity" s)
    Double/NEGATIVE_INFINITY

    (= "NaN" s)
    Double/NaN

    (= "false" s)
    false

    (= "true" s)
    true

    :else s))

(defmacro nan->nil
  "Evaluate the `body` and if the result of the evaluation is `Double/NaN` than
  `nil` is returned or if an exception is raised, otherwise the result is."
  [body]
  `(try
     (let [r# ~body]
       (when-not (and (double? r#)
                      (Double/isNaN r#)) r#))
     (catch Exception e#)))
