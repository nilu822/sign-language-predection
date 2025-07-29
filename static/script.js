$(document).ready(function () {
  let streamStarted = false;
  let eventSource = null;
  let pollInterval = null;

  function initEventSource() {
    if (eventSource) {
      eventSource.close();
    }
    eventSource = new EventSource("/text_stream");
    eventSource.onmessage = function (event) {
      if (event.data) {
        $("#predictedText").val(event.data + " ");
      }
    };
    eventSource.onerror = function () {
      eventSource.close();
      if (streamStarted) {
        setTimeout(initEventSource, 3000);
      }
    };
  }

  $("#startCamera").click(function () {
    if (!streamStarted) {
      $("#videoFeed").attr("src", "/video_feed");
      streamStarted = true;
      initEventSource();
    }
  });

  $("#stopCamera").click(function () {
    $("#videoFeed").attr("src", "");
    streamStarted = false;
    $("#predictedText").val("");
    $.post("/reset_text");
    if (eventSource) {
      eventSource.close();
    }
  });

  $("#trainModel").click(function () {
    const mode = $("#modeSelector").val();
    $("#trainModel").prop("disabled", true);
    $("#trainingSection").show();
    $("#trainingProgress").css("width", "0%").text("0%").show();

    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }

    $.post("/train", { mode: mode }, function () {
      pollInterval = setInterval(() => {
        $.get("/training_progress", function (data) {
          const progress = parseInt(data);
          if (!isNaN(progress)) {
            $("#trainingProgress").css("width", progress + "%").text(progress + "%");
            if (progress >= 100) {
              clearInterval(pollInterval);
              pollInterval = null;
              $("#trainModel").prop("disabled", false);
              setTimeout(() => {
                $("#trainingSection").hide();
                $("#trainingProgress").css("width", "0%").text("0%");
              }, 1000);
            }
          }
        });
      }, 1000);
    });
  });

  $("#speakText").click(function () {
    const text = $("#predictedText").val();
    if (text.trim()) {
      const msg = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.speak(msg);
    }
  });

  $("#resetText").click(function () {
    $.post("/reset_text", function (data) {
      $("#predictedText").val(data);
    });
  });

  $("#backspaceText").click(function () {
    $.post("/backspace_text", function (data) {
      $("#predictedText").val(data);
    });
  });

  $("#addGestureForm").submit(function (e) {
    e.preventDefault();
    const word = $("#gestureInput").val().trim();
    if (!word) return;

    const capitalizedWord =
      word.replace(/\W+/g, "").charAt(0).toUpperCase() +
      word.replace(/\W+/g, "").slice(1);
    const $submitBtn = $("#addGestureForm button[type='submit']");
    $submitBtn.prop("disabled", true);

    // const redirectToCapture = () => {
    //   const msgStart = new SpeechSynthesisUtterance(
    //     `Capturing started for ${capitalizedWord}. Make your gesture naturally.`
    //   );

    //   let redirected = false;

    //   const redirectNow = () => {
    //     if (!redirected) {
    //       redirected = true;
    //       console.log("🔁 Redirect triggered");
    //       window.location.href = "/capture_ui?word=" + encodeURIComponent(capitalizedWord);
    //     }

    const redirectToCapture = () => {
      const msgStart = new SpeechSynthesisUtterance(
        `Capturing started for ${capitalizedWord}. Make your gesture naturally.`
      );

      let redirected = false;

      const redirectNow = () => {
        if (!redirected) {
          redirected = true;
          console.log("🔁 Opening capture in new tab...");
          // window.location.href = `/capture_ui?word=${encodeURIComponent(capitalizedWord)}`;
          window.open(`/capture_ui?word=${encodeURIComponent(capitalizedWord)}`, '_blank');
        }
      };

      msgStart.onend = () => {
        console.log("Speech finished, redirecting...");
        // redirectNow();
        openNewTabAndRedirect();
      };

      msgStart.onerror = (event) => {
        console.error('Speech synthesis error:', event);
        redirectNow(); // Attempt to redirect even on error
      };

      window.speechSynthesis.speak(msgStart);

      // Fallback in case onend doesn't fire
      setTimeout(() => {
        if (!redirected) {
          console.log("Speech onend not fired, redirecting via fallback...");
          redirectNow();
        }
      }, 5000); // 5 seconds fallback
  };

  //   msgStart.onend = redirectNow;
  // // Fallback: if onend doesn't fire within 5 seconds
  //   setTimeout(redirectNow, 5000);
  //   window.speechSynthesis.speak(msgStart);
  //   };


    $.post("/check_word_exists", { word: capitalizedWord }, function (response) {
      if (response === "both_exist") {
        const userChoice = confirm(`Word "${capitalizedWord}" exists. Overwrite?`);
        if (!userChoice) {
          alert("Exited without changes.");
          $submitBtn.prop("disabled", false);
          return;
        }
        $.post("/capture", { word: capitalizedWord, choice: "overwrite" }, function () {
          $("#gestureInput").val("");
          $submitBtn.prop("disabled", false);
          redirectToCapture();
        });
      } else {
        $.post("/capture", { word: capitalizedWord }, function () {
          $("#gestureInput").val("");
          $submitBtn.prop("disabled", false);
          redirectToCapture();
        });
      }
    });
  });

  if (window.location.pathname === "/capture_ui") {
    const word = new URLSearchParams(window.location.search).get("word");
    const halfwaySpoken = { value: false };

    const tts = (text) => {
      const msg = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.speak(msg);
    };

    tts(`Tracking hand for class ${word}. Make your gestures naturally.`);

    const poll = setInterval(() => {
      fetch(`/capture_status?word=${word}`)
        .then((res) => res.text())
        .then((status) => {
          if (status === "halfway" && !halfwaySpoken.value) {
            halfwaySpoken.value = true;
            tts("Halfway completed");
          }

          if (status === "done") {
            clearInterval(poll);
            // tts(`Gesture capture for ${word} completed`);
            // setTimeout(() => (window.location.href = "/"), 3000);
            const completionMsg = new SpeechSynthesisUtterance(`Gesture capture for ${word} completed`);
            completionMsg.onend = () => {
                window.location.href = '/'; // Redirect to homepage after speech
            };
            completionMsg.onerror = (e) => {
                console.error('Speech error on completion:', e);
                window.location.href = '/'; // Redirect to homepage even on error
            };
            window.speechSynthesis.speak(completionMsg);
        //     setTimeout(() => {
        //     window.speechSynthesis.cancel(); // Ensure any ongoing speech is stopped
        //     window.close(); // Close the capture tab
        // }, 3000); // 3 seconds delay before closing

      }
          if (status === "interrupted") {
            clearInterval(poll);
            window.speechSynthesis.cancel(); // Stop any ongoing speech
            const msg = new SpeechSynthesisUtterance(`Capture interrupted for ${word}`);
            msg.onend = () => {
                window.location.href = '/'; // Redirect to homepage on interruption
            };
            msg.onerror = (e) => {
                console.error('Speech error on interruption:', e);
                window.location.href = '/'; // Redirect to homepage even on error
            };
            window.speechSynthesis.speak(msg);
          }
        });
      }, 1000);

    document
      .getElementById("cancelCaptureBtn")
      ?.addEventListener("click", function () {
        fetch(`/capture_status?word=${word}&interrupt=1`).then(() => {
          window.speechSynthesis.cancel();
          const msg = new SpeechSynthesisUtterance(`Capture interrupted for ${word}`);
          // window.speechSynthesis.speak(msg);
          // setTimeout(() => (window.location.href = "/"), 3000);
          msg.onend = () => {
              window.location.href = '/'; // Redirect to homepage on interruption
          };
            msg.onerror = (e) => {
              console.error('Speech error on interruption:', e);
              window.location.href = '/'; // Redirect to homepage even on error
            };
          window.speechSynthesis.speak(msg);
        });
      });

      
      // Add keydown listener for 'q' or 'Q'
    // document.addEventListener("keydown", function (event) {
    //   if (event.key === "q" || event.key === "Q") {
    //     fetch(`/capture_status?word=${word}&interrupt=1`).then(() => {
    //       window.speechSynthesis.cancel(); // Stop any ongoing speech
    //       const msg = new SpeechSynthesisUtterance(`Capture interrupted.`);
    //       // window.speechSynthesis.speak(msg);
    //       // setTimeout(() => (window.location.href = "/"), 3000);
    //       msg.onend = () => {
    //           setTimeout(() => (window.location.href = "/"), 100);
    //       };
    //       msg.onerror = (e) => {
    //           console.error('Speech error on interruption:', e);
    //           setTimeout(() => (window.location.href = "/"), 100);
    //       };
    //       window.speechSynthesis.speak(msg);
    //     });
    //   }
    // });
  }

  $("#downloadBtn").click(function () {
    const mode = $("#modeSelector").val();
    window.location.href = `/download_dataset?mode=${mode}`;
  });
});
























// // Improved version of the selected code
// $("#addGestureForm").submit(function (e) {
//   e.preventDefault();
//   const word = $("#gestureInput").val().trim();
//   if (!word) return;

//   const capitalizedWord = word.replace(/\W+/g, '').charAt(0).toUpperCase() + word.replace(/\W+/g, '').slice(1);
//   const $submitBtn = $("#addGestureForm button[type='submit']");
//   $submitBtn.prop("disabled", true);

//   const redirectToCapture = () => {
//     const msgStart = new SpeechSynthesisUtterance(`Capturing started for ${capitalizedWord}. Make your gesture naturally.`);
//     msgStart.onend = () => {
//       window.location.href = "/capture_ui?word=" + encodeURIComponent(capitalizedWord);
//     };
//     window.speechSynthesis.speak(msgStart);
//   };

//   $.post("/check_word_exists", { word: capitalizedWord }, function (response) {
//     if (response === "both_exist") {
//       const userChoice = confirm(`Word "${capitalizedWord}" exists. Overwrite?`);
//       if (!userChoice) {
//         alert("Exited without changes.");
//         $submitBtn.prop("disabled", false);
//         return;
//       }
//       $.post("/capture", { word: capitalizedWord, choice: "overwrite" }, function () {
//         $("#gestureInput").val("");
//         $submitBtn.prop("disabled", false);
//         redirectToCapture();
//       });
//     } else {
//       $.post("/capture", { word: capitalizedWord }, function () {
//         $("#gestureInput").val("");
//         $submitBtn.prop("disabled", false);
//         redirectToCapture();
//       });
//     }
//   });
// });














// $(document).ready(function () {
//   let streamStarted = false;
//   let eventSource = null;

//   function initEventSource() {
//     if (eventSource) {
//       eventSource.close();
//     }
//     eventSource = new EventSource("/text_stream");
//     eventSource.onmessage = function (event) {
//       if (event.data) {
//         $("#predictedText").val(event.data + " ");
//       }
//     };
//     eventSource.onerror = function () {
//       eventSource.close();
//       if (streamStarted) {
//         setTimeout(initEventSource, 3000);
//       }
//     };
//   }

//   initEventSource();

//   $("#startCamera").click(function () {
//     if (!streamStarted) {
//       $("#videoFeed").attr("src", "/video_feed");
//       streamStarted = true;
//       initEventSource();
//     }
//   });

//   $("#stopCamera").click(function () {
//     $("#videoFeed").attr("src", "");
//     streamStarted = false;
//     $("#predictedText").val("");
//     $.post("/reset_text");
//     if (eventSource) {
//       eventSource.close();
//     }
//   });

//   $("#trainModel").click(function () {
//     const mode = $("#modeSelector").val();
//     $("#trainingSection").show();
//     $("#trainingProgress").css("width", "0%").text("0%").show();

//     $.post("/train", { mode: mode }, function () {
//       const pollInterval = setInterval(() => {
//         $.get("/training_progress", function (data) {
//           const progress = parseInt(data);
//           if (!isNaN(progress)) {
//             $("#trainingProgress").css("width", progress + "%").text(progress + "%");
//             if (progress >= 100) {
//               clearInterval(pollInterval);
//               setTimeout(() => {
//                 $("#trainingProgress").hide();
//               }, 1000);
//             }
//           }
//         });
//       }, 1000);
//     });
//   });

//   $("#speakText").click(function () {
//     const text = $("#predictedText").val();
//     if (text.trim()) {
//       const msg = new SpeechSynthesisUtterance(text);
//       window.speechSynthesis.speak(msg);
//     }
//   });

//   $("#resetText").click(function () {
//     $.post("/reset_text", function (data) {
//       $("#predictedText").val(data);
//     });
//   });

//   $("#backspaceText").click(function () {
//     $.post("/backspace_text", function (data) {
//       $("#predictedText").val(data);
//     });
//   });

//   $("#addGestureForm").submit(function (e) {
//     e.preventDefault();
//     const word = $("#gestureInput").val();
//     $.post("/capture", { word: word }, function () {
//       alert(`Gesture "${word}" captured.`);
//       $("#gestureInput").val("");
//     });
//   });

//   $("#downloadBtn").click(function () {
//     const mode = $("#modeSelector").val();
//     window.location.href = `/download_dataset?mode=${mode}`;
//   });
// });




















// this is working ---- having fake bar --- 100% working 
// $(document).ready(function () {
//   let streamStarted = false;
//   let eventSource = null; // Declare eventSource outside to manage reconnection

//   // for hover on start camera button --
//   document.getElementById("startCamera").addEventListener("click", () => {
//   const camFrame = document.querySelector(".camera-frame");
//   camFrame.classList.add("active");
//   setTimeout(() => camFrame.classList.remove("active"), 600);
// });


//   // Function to initialize EventSource
//   function initEventSource() {
//     if (eventSource) {
//       eventSource.close(); // Close existing connection if any
//       console.log("Closed existing EventSource connection.");
//     }
//     eventSource = new EventSource("/text_stream");
//     console.log("Attempting to open new EventSource connection...");

//     eventSource.onmessage = function (event) {
//       if (event.data) {
//         $("#predictedText").val(event.data + " ");
//       }
//     };

//     eventSource.onerror = function (err) { // ADDED: Error handling for EventSource
//       console.error("EventSource failed:", err);
//       eventSource.close(); // Close the current broken connection
//       // Attempt to reconnect after a delay if the camera feed is active
//       if (streamStarted) { // Only reconnect if camera is active
//         setTimeout(() => {
//           console.log("Attempting to reconnect to /text_stream...");
//           initEventSource(); // Re-initialize EventSource
//         }, 3000); // Try reconnecting after 3 seconds
//       } else {
//           console.log("EventSource not reconnected: Camera is off.");
//       }
//     };
//   }

//   // Initialize EventSource on page load
//   initEventSource();


//   $("#darkModeToggle").click(function () {
//     $("body").toggleClass("dark-mode light-mode");
//   });

//   $("#startCamera").click(function () {
//     if (!streamStarted) {
//       $("#videoFeed").attr("src", "/video_feed");
//       streamStarted = true;
//       // Ensure EventSource is active when camera starts
//       initEventSource(); // Re-initialize in case it died or wasn't active
//     }
//   });

//   // MODIFIED: Added logic to clear text area and reset backend text
//   $("#stopCamera").click(function () {
//     $("#videoFeed").attr("src", "");
//     streamStarted = false;
//     $("#predictedText").val(""); // Clear the text area on stop
//     // Send a request to the backend to reset its text
//     $.post("/reset_text", function(data) {
//         console.log("Backend text reset after stopping feed:", data);
//     });
//     // Close EventSource when camera is off
//     if (eventSource) {
//       eventSource.close();
//       console.log("EventSource closed due to camera stop.");
//     }
//   });

//   $("#trainModel").click(function () {
//     const mode = $("#modeSelector").val();
//     $("#trainingSection").show();
//     $("#trainingProgress").css("width", "10%");


//     $.post("/train", { mode: mode }, function () {
//       let progress = 10;
//       const interval = setInterval(() => {
//         if (progress >= 100) {
//           clearInterval(interval);
//           $("#trainingProgress").text("Training Complete!");
//         } else {
//           progress += 5;
//           $("#trainingProgress").css("width", progress + "%");
//         }
//       }, 500);
//     });
//   });



//   $("#speakText").click(function () {
//     const text = $("#predictedText").val();
//     if (text.trim()) {
//       const msg = new SpeechSynthesisUtterance(text);
//       window.speechSynthesis.speak(msg);
//     }
//   });

//   $("#resetText").click(function () {
//     $.post("/reset_text", function (data) {
//       $("#predictedText").val(data);
//     });
//   });

//   $("#backspaceText").click(function () {
//     const text = $("#predictedText").val().trim();
//     const words = text.split(" ");
//     words.pop();
//     $("#predictedText").val(words.join(" ") + " ");

//     $.post("/backspace_text", function (data) {
//       $("#predictedText").val(data);
//     });
//   });

//   $("#addGestureForm").submit(function (e) {
//     e.preventDefault();
//     const word = $("#gestureInput").val();
//     $.post("/capture", { word: word }, function () {
//       alert(`Gesture "${word}" captured.`);
//       $("#gestureInput").val("");
//     });
//   });

//   $("#downloadBtn").click(function () {
//     const mode = $("#modeSelector").val();
//     window.location.href = `/download_dataset?mode=${mode}`;
//   });
// });


































// // ========================== script.js ==========================
// $(document).ready(function () {
//   let streamStarted = false;
//   let eventSource = null; // Declare eventSource outside to manage reconnection

//   // Function to initialize EventSource
//   function initEventSource() {
//     if (eventSource) {
//       eventSource.close(); // Close existing connection if any
//       console.log("Closed existing EventSource connection.");
//     }
//     eventSource = new EventSource("/text_stream");
//     console.log("Attempting to open new EventSource connection...");

//     eventSource.onmessage = function (event) {
//       if (event.data) {
//         $("#predictedText").val(event.data + " ");
//       }
//     };

//     eventSource.onerror = function (err) { // ADDED: Error handling for EventSource
//       console.error("EventSource failed:", err);
//       eventSource.close(); // Close the current broken connection
//       // Attempt to reconnect after a delay if the camera feed is active
//       if (streamStarted) { // Only reconnect if camera is active
//         setTimeout(() => {
//           console.log("Attempting to reconnect to /text_stream...");
//           initEventSource(); // Re-initialize EventSource
//         }, 3000); // Try reconnecting after 3 seconds
//       } else {
//           console.log("EventSource not reconnected: Camera is off.");
//       }
//     };
//   }

//   // Initialize EventSource on page load
//   initEventSource();


//   $("#darkModeToggle").click(function () {
//     $("body").toggleClass("dark-mode light-mode");
//   });

//   $("#startCamera").click(function () {
//     if (!streamStarted) {
//       $("#videoFeed").attr("src", "/video_feed");
//       streamStarted = true;
//       // Ensure EventSource is active when camera starts
//       initEventSource(); // Re-initialize in case it died or wasn't active
//     }
//   });

//   $("#stopCamera").click(function () {
//     $("#videoFeed").attr("src", "");
//     streamStarted = false;
//     // Close EventSource when camera is off
//     if (eventSource) {
//       eventSource.close();
//       console.log("EventSource closed due to camera stop.");
//     }
//   });

//   $("#trainModel").click(function () {
//     const mode = $("#modeSelector").val();
//     $("#trainingSection").show();
//     $("#trainingProgress").css("width", "10%");

//     $.post("/train", { mode: mode }, function () {
//       let progress = 10;
//       const interval = setInterval(() => {
//         if (progress >= 100) {
//           clearInterval(interval);
//           $("#trainingProgress").text("Training Complete!");
//         } else {
//           progress += 5;
//           $("#trainingProgress").css("width", progress + "%");
//         }
//       }, 500);
//     });
//   });

//   $("#speakText").click(function () {
//     const text = $("#predictedText").val();
//     if (text.trim()) {
//       const msg = new SpeechSynthesisUtterance(text);
//       window.speechSynthesis.speak(msg);
//     }
//   });

//   $("#resetText").click(function () {
//     // UPDATED LOGIC: Send POST to backend and update UI from response
//     $.post("/reset_text", function (data) {
//       $("#predictedText").val(data);
//     });
//   });

//   $("#backspaceText").click(function () {
//     // UPDATED LOGIC: Perform client-side visual update, then send POST and update UI from response
//     // This provides immediate feedback while waiting for backend confirmation
//     const text = $("#predictedText").val().trim();
//     const words = text.split(" ");
//     words.pop();
//     $("#predictedText").val(words.join(" ") + " "); // Immediate client-side update

//     $.post("/backspace_text", function (data) {
//       $("#predictedText").val(data); // Final update from backend's authoritative text
//     });
//   });

//   $("#addGestureForm").submit(function (e) {
//     e.preventDefault();
//     const word = $("#gestureInput").val();
//     $.post("/capture", { word: word }, function () {
//       alert(`Gesture "${word}" captured.`);
//       $("#gestureInput").val("");
//     });
//   });

//   $("#downloadBtn").click(function () {
//     const mode = $("#modeSelector").val();
//     window.location.href = `/download_dataset?mode=${mode}`;
//   });

//   // Original: Optional: Live update text area from prediction frame
//   // This is now handled by initEventSource() and its onmessage/onerror
// });
