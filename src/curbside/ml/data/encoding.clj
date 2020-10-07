(ns curbside.ml.data.encoding
  "Feature encoding consists of representing features in an alternative
  representation to facilitate learning. This is mainly used to represent
  categorical features as numbers to input them to a machine learning model."
  (:require
   [clojure.spec.alpha :as s]
   [curbside.ml.utils.spec :as spec-utils]))

(s/def ::feature keyword?)
(s/def ::feature-value (s/or :number number? :string string?))

(s/def ::one-hot-vector
  (s/cat :zeros (s/* zero?)
         :one #{1}
         :zeros (s/* zero?)))

(s/def ::one-hot-vectors
  (s/and
   (s/map-of ::feature-value ::one-hot-vector)
   #(apply = (map count (vals %)))))

(s/def :one-hot-encoding/type #{:one-hot})

(s/def ::one-hot-encoding
  (s/keys :req-un [:encoding-fn/type
                   ::one-hot-vectors]))

(s/def ::encoding-fn ::one-hot-encoding) ;; Only this encoding is supported for now

(s/def ::features (s/map-of ::feature ::encoding-fn))

(s/def ::dataset-encoding (s/keys :req-un [::features]))

(defn- identity-matrix
  "Returns an identity matrix of size `n`."
  [n]
  (for [i (range n)]
    (vec (concat (repeat i 0)
                 [1]
                 (repeat (- n i 1) 0)))))

(defn create-one-hot-encoding
  [vals]
  {:post [(spec-utils/check ::one-hot-encoding %)]}
  (let [distinct-vals (distinct vals)]
    {:type :one-hot
     :one-hot-vectors (zipmap distinct-vals
                              (identity-matrix (count distinct-vals)))}))

(defn- one-hot-encode-value
  [{:keys [one-hot-vectors] :as _encoding-fn} value]
  (get one-hot-vectors value))

(defn- encode-value
  [encoding-fn value]
  (case (:type encoding-fn)
    :one-hot (one-hot-encode-value encoding-fn value)))

(defn encode-feature-map
  [dataset-encoding feature-map]
  {:pre [(spec-utils/check ::dataset-encoding dataset-encoding)]}
  (reduce (fn [feature-map [feature encoding-fn]]
            (update feature-map feature (partial encode-value encoding-fn)))
          feature-map
          (:features dataset-encoding)))
