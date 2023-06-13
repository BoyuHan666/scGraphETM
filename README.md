# scGraphETM

### Get start

```
git init
git remote add origin git@github.com:BoyuHan666/scGraphETM.git
```

### Develop on a feature branch

```
git checkout master
git pull origin master
git checkout -b my-branch
# Write some code
```

### Commit and create pull request

```
git add <files>
git commit -m '<message>'
git pull --rebase origin master # You might need to resolve merge conflicts after this command
git push -u origin HEAD
```

### Update PR after writing some more code & addressing comments

Go to the PR, edit title and description as needed, this will be the commit message once merged into master!

Lastly, merge the PR

The remote feature branch will be deletely automatically. You can also delete the local feature branch.
