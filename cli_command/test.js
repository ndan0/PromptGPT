fetch('http://127.0.0.1:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    instances: [
      {
        text: 'What is the weather like?'
      }
    ]
  })
})
  .then(response => response.json())
  .then(data => {
    console.log(data);
    // Handle the response data as needed
  })
  .catch(error => {
    console.error('Error:', error);
    // Handle any errors that occur
  });


