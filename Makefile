SRC_DIR = src
TEST_SRC_DIR = test

.PHONY: clean
clean:
	rm -rf $(SRC_DIR)
	rm -rf $(TEST_SRC_DIR)

.PHONY: lint
lint:
	lein cljfmt

.PHONY: lint-fix
lint-fix:
	lein cljfmt fix

.PHONY: code-quality-check
code-quality-check:
	lein eastwood

.PHONY: test
test:
	lein test

.PHONY: test-jenkins
test-jenkins:
	lein trampoline with-profile ci test2junit
