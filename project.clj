(defproject com.curbside/curbside-clojure-ml "4.1.6"
  :description "Library for ML model training and serving."
  :url "http://github.com/RakutenReady/curbside-clojure-ml"
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [org.clojure/core.async "1.0.567"]
                 [org.clojure/data.csv "0.1.4"]
                 [clj-time "0.15.2"]
                 [medley "1.2.0"]
                 [org.clojure/math.combinatorics "0.1.6"]
                 [org.clojure/math.numeric-tower "0.0.4"]
                 [com.climate/claypoole "1.1.4"]

                 ;; Conjure for mocking/stubbing
                 [org.clojars.runa/conjure "2.1.3"]

                 ;; Spec helper
                 [expound "0.7.2"]

                 ;; weka ML
                 [nz.ac.waikato.cms.weka/weka-dev "3.9.3"]

                 ;; SVM
                 [de.bwaldvogel/liblinear "2.30"]
                 [tw.edu.ntu.csie/libsvm "3.24"]

                 ;; Graphviz
                 [guru.nidi/graphviz-java "0.11.0"]

                 ;; xgboost
                 [ml.dmlc/xgboost4j "0.90"]

                 ;; Apache Commons Math3
                 [org.apache.commons/commons-math3 "3.6.1"]

                 ;; Logging
                 [org.clojure/tools.logging "1.0.0"]
                 [org.slf4j/slf4j-log4j12 "1.7.30"]

                 ;;benchmarking
                 [net.totakke/libra "0.1.1"]]

  :profiles {:uberjar {:aot :all :global-vars {*assert* false}}
             :ci {:plugins [[test2junit "1.3.3"]]}
             :test {:resource-paths ["test-resources"]
                    :dependencies [[org.clojure/test.check "1.1.0"]]}
             :dev [:test
                   {:dependencies [[criterium "0.4.6"]]
                    :global-vars {*warn-on-reflection* true}}]}

  :plugins [[com.gfredericks/lein-how-to-ns "0.2.7"]
            [lein-ancient "0.6.15"]
            [jonase/eastwood "0.3.11"]
            [lein-cljfmt "0.6.8"]
            [net.totakke/lein-libra "0.1.2"]]

  :how-to-ns {:require-docstring? false
              :sort-clauses? true
              :allow-refer-all? false
              :allow-extra-clauses? false
              :align-clauses? false
              :import-square-brackets? false}

  :cljfmt {:indents {instrumenting [[:block 1]]
                     mocking [[:block 1]]
                     stubbing [[:block 1]]
                     mocking-private [[:block 1]]
                     stubbing-private [[:block 1]]
                     timed [[:block 2]]
                     for-all [[:block 1]]}}

  :deploy-repositories [["releases" {:url "https://maven.pkg.github.com/RakutenReady/curbside-clojure-ml"
                                    :username :env/github_actor
                                    :password :env/github_token
                                    :sign-releases false}]]

  :test2junit-output-dir "test-reports"

  :eastwood {:exclude-linters [:unlimited-use ;; used in tests
                               :def-in-def ;; false positives from stubbing
                               :unused-fn-args ;; many false positives -- https://github.com/jonase/eastwood/issues/21
                               :deprecations]
             :add-linters [:unused-locals
                           :implicit-dependencies
                           :local-shadows-var
                           :misplaced-docstrings
                           :suspicious-expression
                           :suspicious-test
                           :unused-private-vars
                           :unused-ret-vals
                           :unused-ret-vals-in-try]}

  :aliases {"fix" ["do" ["cljfmt" "fix"] ["how-to-ns" "fix"]]
            "check" ["do" ["cljfmt" "check"] ["how-to-ns" "check"]]}

  :jvm-opts ["-XX:-OmitStackTraceInFastThrow"])
