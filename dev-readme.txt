Testing and merging:
====================

1. Locally:

pixi run -e lint lint

pixi run -e test test

pixi run build

pixi run -e doc build-doc

Commit to a dev branch.


2. CI:

Push dev branch to Github.

Create pull request to trigger CI tests.

# Merge locally, as a workaround for https://github.com/orgs/community/discussions/5524,
# "PR's "Rebase and Merge" should not alter commits if the head branch is already on top of the main one".
printf 'Fast-forwarding main to dev: '; git merge-base --is-ancestor main dev && git branch -f main dev && echo OK || echo ERROR

git push origin main:main

Delete dev branch in pull request, for easier creation of the next pull request.

