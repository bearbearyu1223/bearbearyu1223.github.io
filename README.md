# Welcome to Han’s Log 👋 
Hi, this is Han. I’m documenting my learning notes in this blog. Besides, I’m a product manager at Apple AI/ML. You can connect me via [![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/han-yu-goirish/). Below, I also document the process re how to set up local dev env and delpoy Jekyll site to GitHub pages. 


# 🚀 Jekyll on MacOS + GitHub Pages Deployment

This guide walks you through setting up your MacOS dev env to build and run Jekyll locally, and deploy your Jekyll site to GitHub Pages.

---

## ✅ Prerequisites

- macOS 
- [Homebrew](https://brew.sh/)
- GitHub account
- Git installed and configured

---

## 🧰 Setup Instructions

### 1. Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### 2. Install Ruby via rbenv 
Ruby 3.3.x is not compatible with Jekyll 4.3.x. Use Ruby 3.2.2 instead. Follow the steps below
```bash
brew install rbenv ruby-build
echo 'eval "$(rbenv init -)"' >> ~/.zshrc
source ~/.zshrc
rbenv install 3.2.2
rbenv global 3.2.2
```

Verify installation 
```bash
ruby -v  # should show ruby 3.2.2
```

### 3. Install Jekyll and Bundler 
```bash
gem install bundler jekyll
```

### 4. Create a New Jekyll Site 
```bash
jekyll new my-awesome-site
cd my-awesome-site
bundle install
```
To preview locally 
```bash
bundle exec jekyll serve
```
Visit http://localhost:4000 in your browser.

## 🚀 Deploying to GitHub Pages
### 1. Create a GitHub Repository
Name it: yourusername.github.io

Replace yourusername with your actual GitHub username.

### 2. Initialize Git and Push Site
```bash
cd my-awesome-site
git init
git remote add origin https://github.com/yourusername/yourusername.github.io.git
git add .
git commit -m "Initial commit"
git push -u origin master  # or `main`, depending on default branch
```
### 3. Configure _config.yml
In your Jekyll site directory:
```yaml
# _config.yml
url: "https://yourusername.github.io"
```
### 4. Enable GitHub Pages
Go to your repo on GitHub

Settings → Pages

Source: set to Deploy from a branch, use main or master, and / as the folder.

GitHub will serve your site at:
👉 https://yourusername.github.io