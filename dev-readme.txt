Testing and merging:
====================

1. Locally:

pixi run -e lint lint

pixi run -e test test

pixi run -e doc build-doc

Commit to a dev branch.

pixi run build


2. CI:

Push dev branch to Github.

Create pull request to trigger CI tests.

# Merge locally, as a workaround for https://github.com/orgs/community/discussions/5524,
# "PR's "Rebase and Merge" should not alter commits if the head branch is already on top of the main one".
printf 'Fast-forwarding main to dev: '; git merge-base --is-ancestor main dev && git branch -f main dev && echo OK || echo ERROR

git push origin main:main
# By this, The Github repo setting "Automatically delete head branches"
# deletes the dev branch in the pull request, for easier creation of the next pull request from dev.

