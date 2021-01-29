(ns curbside.ml.models-bench
  (:require [libra.bench :refer :all]
          [libra.criterium :as c]
            [clojure.java.io :as io]
          [curbside.ml.models :as models]
            [clojure.edn :as edn]))

(defbench infer-xgboost-bench
  (let [model (models/load :xgboost (.getPath (io/as-file (io/resource "models/eta/regression-test/5659.xgb"))))
        factors (edn/read-string (slurp (io/resource "models/eta/regression-test/5659.scale.edn")))
        config (edn/read-string (slurp (io/resource "models/eta/regression-test/5659.config.edn")))
        feature-map {:average-approaching-speed-4 0.0
                     :cosine-angle -0.636894300348137
                     :site-lng -82.3596632480621
                     :time-of-day 16
                     :approaching-speed 0.0
                     :site-lat 34.8413076694805
                     :euclidean-distance 1769004.9423246796
                     :locations [{:lat 48.417492, :lng -71.151178, :ts "2019-02-18T15:35:07Z"}]
                     :vertical-distance -1363816.3653824695
                     :line-distance 40875.06331285569
                     :sine-angle -0.7709511334605178
                     :lat 48.34444
                     :horizontal-distance -1126669.165054273
                     :lng -71.204523}]
    (is (c/bench (models/infer :xgboost :regression model
                               (get-in config [:features-selection :selected-features])
                               (get-in config [:train-models :hyperparameters])
                               feature-map
                               :scaling-factors factors
                               :feature-scaling-fns []
                               :label-scaling-fns [:log10]
                               :dataset-encoding nil)))))
