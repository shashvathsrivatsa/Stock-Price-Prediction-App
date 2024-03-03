function getData () {
    console.clear();
    let symbol = document.getElementById('symbol-input').value;
    // fetch('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=' + symbol + '&interval=1min&apikey=LI6KQURHNB85O9B4')
    fetch('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol=' + symbol + '&apikey=LI6KQURHNB85O9B4')
    .then(response => response.json())
    .then(response => {

        console.log(response);

        let time = [];
        let close = [];
        for (let date in response['Time Series (Daily)']) {
            const data = response['Time Series (Daily)'][date];
            time.unshift(date);
            close.unshift(parseFloat(data['4. close']));
        }
        time = time.slice(-200);
        close = close.slice(-200);

        //  ======================  PREDICTION MODEL (LSTM)  ======================  //
        
        // create and compile model
        console.log('=======  STATUS  =======');
        console.log('creating model');
        let denormalizedPredictedValues;
        const model = tf.sequential();
        const sequence_length = 10
        model.add(tf.layers.lstm({
            units: 50,
            activation: 'relu',
            inputShape: [sequence_length, 1],
            returnSequences: true
        }));
        model.add(tf.layers.lstm({
            units: 50,
            itivation: 'relu'
        }));
        model.add(tf.layers.dense({
            units: 1
        }));
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });

        //  normalize data
        console.log('normalizing data');
        const closeMax = Math.max(...close);
        const closeMin = Math.min(...close);
        const normalizedClose = close.map(price => (price - closeMin) / (closeMax - closeMin));

        //  split data into training and testing sets
        console.log('splitting data');
        const closeSplitIndex = Math.floor(normalizedClose.length * 0.7);
        const trainClosePrices = normalizedClose.slice(0, closeSplitIndex);
        const verifyClosePrices = normalizedClose.slice(closeSplitIndex);

        //  format data for training
        console.log('formatting data');
        const X_train_lstm = [];
        const y_train_lstm = [];
        for (let i = 0; i < trainClosePrices.length - sequence_length; i++) {
            const inputSlice = trainClosePrices.slice(i, i + sequence_length);
            const target = trainClosePrices[i + sequence_length];
            X_train_lstm.push(inputSlice.map(price => [price]));
            y_train_lstm.push(target);
        }
        const X_train = tf.tensor3d(X_train_lstm, [X_train_lstm.length, sequence_length, 1]);
        const y_train = tf.tensor2d(y_train_lstm, [y_train_lstm.length, 1]);

        //  train model
        console.log('training model');
        model.fit(X_train, y_train, {
            epochs: 50,
            batchSize: 8,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch: ${epoch + 1}/50 \nLoss: ${logs.loss}`);
                }
            }
        }).then(() => {

        //  predictions
            console.log('predicting');
            const X_verify_lstm = [];
            const y_verify_lstm = [];
            for (let i = 0; i < verifyClosePrices.length - sequence_length; i++) {
                const inputSlice = verifyClosePrices.slice(i, i + sequence_length);
                const target = verifyClosePrices[i + sequence_length];
                X_verify_lstm.push(inputSlice.map(price => [price]));
                y_verify_lstm.push(target);
            }
            const X_verify = tf.tensor3d(X_verify_lstm, [X_verify_lstm.length, sequence_length, 1]);
            const y_verify = tf.tensor2d(y_verify_lstm, [y_verify_lstm.length, 1]);
            
            // verify predictions
            console.log('verifying');
            const predictedTensor = model.predict(X_verify);
            const predictedValuesTensor = predictedTensor.dataSync();
            const predictedValues = Array.from(predictedValuesTensor);

            //  denormalize predicted values
            console.log('denormalizing predicted values');
            denormalizedPredictedValues = predictedValues.map(value => (value * (closeMax - closeMin)) + closeMin);
            console.log('========================');
            console.log("Predicted Close Prices:", denormalizedPredictedValues);
            console.log("Actual Close Prices:", y_verify.dataSync().map(value => (value * (closeMax - closeMin)) + closeMin));

        }).then(() => {

            //  ======================  PLOTLY  ======================  //

            let min;
            let max;
            if (Math.min(...close) <= Math.min(...denormalizedPredictedValues)) {
                min = Math.min(...close);
            } else {
                min = Math.min(...denormalizedPredictedValues);
            }
            if (Math.max(...close) >= Math.max(...denormalizedPredictedValues)) {
                max = Math.max(...close);
            } else {
                max = Math.max(...denormalizedPredictedValues);
            }
            const range = max - min;

            if (min - 0.1 * range < 0) {
                var lowerBound = 0
            } else {
                var lowerBound = min - 0.1 * range
            };
            const upperBound = max + 0.1 * range;

            graph = document.getElementById('graph');

            timeSplitIndex = Math.floor(time.length * 0.3) - sequence_length;

            let data = [
                {
                    x: time,
                    y: close,
                    line: {
                        color: '',
                        hoverline: {
                            color: 'blue',
                        }
                    },
                    hovertemplate: '$%{y} <extra></extra>',
                    name: 'Actual Close Prices'
                },
                {
                    x: time.slice(-timeSplitIndex),
                    y: denormalizedPredictedValues,
                    line: {
                        color: 'red',
                        hoverline: {
                            color: 'red',
                        }
                    },
                    hovertemplate: '$%{y} <extra></extra>',
                    name: 'Predicted Close Prices',
                    mode: 'lines'
                }
            ];

            var layout = {
                font: {
                    color: '#fff',
                    family: 'Avenir',
                },
                title: symbol.toUpperCase(),
                plot_bgcolor: '#181A36',
                paper_bgcolor: '#181A36',
                dragmode: 'pan',
                xaxis: {
                    title: {
                        text: 'Time',
                    },
                    color: '#ccc',
                    showgrid: true,
                    showline: true,
                },
                yaxis: {
                    title: {
                        text: 'Price',
                        standoff: 20,
                    },

                    color: '#ccc',
                    showgrid: true,
                    dtick: (upperBound - lowerBound) / 10,
                    tick0: 0,
                    showline: true,
                    range: [lowerBound, upperBound],
                },
                hovermode: 'x',
            };

            var config = {
                displaylogo: false,
                displayModeBar: false,
                scrollZoom: true,
            }

            Plotly.newPlot(graph, data, layout, config)

        });

    })
}
