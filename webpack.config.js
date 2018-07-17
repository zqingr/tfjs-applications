const path = require('path');

module.exports = function (env) {
  return {
    mode: 'production',
    context: path.join(process.cwd(), 'src'),
    target: 'node',
    entry: {
      index: './index.ts'
    },
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: '[name].js',
      libraryTarget: "commonjs"
    },
    resolve: {
      // Add `.ts` and `.tsx` as a resolvable extension.
      extensions: [".ts", ".tsx", ".js"]
    },
    module: {
      rules: [
      {
        test: /\.(ts)$/,
        enforce: "pre",
        exclude: /node_modules/,
        loader: "eslint-loader",
        options: {
          fix: true
        }
      }, 
      {
        test: /\.ts$/,
        exclude: /node_modules/,
        loader: 'ts-loader',
        options: {
          appendTsSuffixTo: [/\.vue$/, /\.ts$/],
          transpileOnly: true
        }
      },
      {
        test: /\.html$/,
        loader: 'raw-loader',
        exclude: ['./src/index.html']
      },
      {
        test: /\.(jpg|jpeg|gif|png)$/,
        exclude: /node_modules/,
        loader: [
          `url-loader?limit=4112&publicPath=/`
        ]
      },
      {
        test: /\.css$/,
        exclude: /node_modules/,
        use: [
          'style-loader',
          'css-loader'
        ]
      }]
    },
    externals: {
      '@tensorflow/tfjs': '@tensorflow/tfjs',
      '@tensorflow/tfjs-node': '@tensorflow/tfjs-node',
      'canvas': 'canvas'
    }
  }
}