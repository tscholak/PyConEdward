# My `reveal.js` Presentation Template

This is based on [reveal.js](https://github.com/hakimel/reveal.js), a framework for creating presentations using HTML, CSS, and Markdown.

## Steps to Make Use of This

1. Clone the repository:
```
$ git clone --recursive https://github.com/tscholak/presentation_template.git new_presentation
```

2. Set up npm:
```
$ cd new_presentation
$ npm init
```
npm will ask a couple of questions. If unsure, just go with the standard answers.

3. Install grunt and its components:
```
$ npm install grunt grunt-contrib-qunit grunt-contrib-jshint grunt-contrib-cssmin \
	grunt-contrib-uglify grunt-contrib-watch grunt-sass grunt-contrib-connect \
	grunt-autoprefixer grunt-zip --save-dev
```

4. Build `reveal.js` and the custom `css` files from their sources:
```
$ grunt --force
```

5. Preview the presentation in a browser:
```
$ grunt serve
```

