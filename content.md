external markdown slide

## Code Example

$L$ Lorem $x^2$ [ipsum](https://en.wikipedia.org/wiki/Lorem_ipsum#Example_text).

$f(x) = \int\_{-\infty}^\infty \hat f(\xi)\,\mathrm{e}^{2 \pi i \xi x} \,\mathrm{d}\xi$

```javascript
function linkify( selector ) {
  if( supports3DTransforms ) {

    var nodes = document.querySelectorAll( selector );

    for( var i = 0, len = nodes.length; i &lt; len; i++ ) {
      var node = nodes[i];

      if( !node.className ) {
        node.className += ' roll';
      }
    }
  }
}
```

Note:
Here are some speaker notes.

This can *contain markdown*.