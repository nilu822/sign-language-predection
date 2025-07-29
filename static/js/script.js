// ========================== script.js ==========================
$(document).ready(function () {
  let streamStarted = false;
  let eventSource = null; // Declare eventSource outside to manage reconnection

  // Function to initialize EventSource
  function initEventSource() {
    if (eventSource) {
      eventSource.close(); // Close existing connection if any
      console.log("Closed existing EventSource connection.");
    }
    eventSource = new EventSource("/text_stream");
    console.log("Attempting to open new EventSource connection...");

    eventSource.onmessage = function (event) {
      if (event.data) {
        $("#predictedText").val(event.data + " ");
      }
    };

    eventSource.onerror = function (err) { // ADDED: Error handling for EventSource
      console.error("EventSource failed:", err);
      eventSource.close(); // Close the current broken connection
      // Attempt to reconnect after a delay if the camera feed is active
      if (streamStarted) { // Only reconnect if camera is active
        setTimeout(() => {
          console.log("Attempting to reconnect to /text_stream...");
          initEventSource(); // Re-initialize EventSource
        }, 3000); // Try reconnecting after 3 seconds
      } else {
          console.log("EventSource not reconnected: Camera is off.");
      }
    };
  }

  // Initialize EventSource on page load
  initEventSource();


  $("#darkModeToggle").click(function () {
    $("body").toggleClass("dark-mode light-mode");
  });

  $("#startCamera").click(function () {
    if (!streamStarted) {
      $("#videoFeed").attr("src", "/video_feed");
      streamStarted = true;
      // Ensure EventSource is active when camera starts
      initEventSource(); // Re-initialize in case it died or wasn't active
    }
  });

  $("#stopCamera").click(function () {
    $("#videoFeed").attr("src", "");
    streamStarted = false;
    // Close EventSource when camera is off
    if (eventSource) {
      eventSource.close();
      console.log("EventSource closed due to camera stop.");
    }
  });

  $("#trainModel").click(function () {
    const mode = $("#modeSelector").val();
    $("#trainingSection").show();
    $("#trainingProgress").css("width", "10%");

    $.post("/train", { mode: mode }, function () {
      let progress = 10;
      const interval = setInterval(() => {
        if (progress >= 100) {
          clearInterval(interval);
          $("#trainingProgress").text("Training Complete!");
        } else {
          progress += 5;
          $("#trainingProgress").css("width", progress + "%");
        }
      }, 500);
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
    // ORIGINAL LOGIC: Send POST and update text area from response
    $.post("/reset_text", function (data) {
      $("#predictedText").val(data);
    });
  });

  $("#backspaceText").click(function () {
    // ORIGINAL LOGIC: Send POST and update text area from response
    const text = $("#predictedText").val().trim();
    const words = text.split(" ");
    words.pop();
    $("#predictedText").val(words.join(" ") + " ");
    $.post("/backspace_text", function (data) {
      $("#predictedText").val(data);
    });
  });

  $("#addGestureForm").submit(function (e) {
    e.preventDefault();
    const word = $("#gestureInput").val();
    $.post("/capture", { word: word }, function () {                //ORIGINAL LOGIC: No 'mode' in payload
      alert(`Gesture "${word}" captured.`);                   // ORIGINAL LOGIC: Alert message
      $("#gestureInput").val("");
    });
  });

  $("#downloadBtn").click(function () {
    const mode = $("#modeSelector").val();
    window.location.href = `/download_dataset?mode=${mode}`;
  });

  // Optional: Live update text area from prediction frame
  // This is now handled by initEventSource() and its onmessage/onerror
});















//orignal ----
// console.log("[JS] script.js loaded");

// $(document).ready(function () {
//   let streamStarted = false;
//   let textSyncPaused = false;
//   let lastSyncedText = "";

//   function showToast(message) {
//     const toast = $("<div class='custom-toast'></div>").text(message);
//     toast.appendTo("body");
//     setTimeout(() => toast.fadeOut(500, () => toast.remove()), 3000);
//   }

//   $("#darkModeToggle").click(function () {
//     $("body").toggleClass("dark-mode light-mode");
//   });

//   $("#startCamera").click(function () {
//     if (!streamStarted) {
//       $("#videoFeed").attr("src", "/video_feed");
//       streamStarted = true;
//     }
//   });

//   $("#stopCamera").click(function () {
//     $("#videoFeed").attr("src", "");
//     streamStarted = false;
//   });

//   $("#trainModel").click(function () {
//     const mode = $("#modeSelector").val();
//     $("#trainModel").prop("disabled", true);
//     $("#trainingSection").show();
//     $("#trainingProgress").css("width", "0%").text("Starting...");

//     const tts = new SpeechSynthesisUtterance("Training started");
//     window.speechSynthesis.speak(tts);

//     $.post("/train", { mode: mode }, function () {
//       let interval = setInterval(() => {
//         $.get("/progress", function (epoch) {
//           const maxEpoch = 30;
//           const percent = Math.min((parseInt(epoch) / maxEpoch) * 100, 100);
//           $("#trainingProgress").css("width", percent + "%");
//           $("#trainingProgress").text(`Epoch ${epoch}/${maxEpoch}`);

//           if (parseInt(epoch) >= maxEpoch) {
//             clearInterval(interval);
//             $("#trainingProgress").text("✅ Training Complete!");
//             showToast("🎉 Training completed successfully!");
//             const completeTTS = new SpeechSynthesisUtterance("Training completed");
//             window.speechSynthesis.speak(completeTTS);
//             $("#trainModel").prop("disabled", false);
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
//     console.log("[JS] Reset button clicked");
//     textSyncPaused = true;
//     $.post("/reset_text", function (updatedText) {
//       console.log("[JS] Reset POST response:", updatedText);
//       $("#predictedText").val(updatedText.trim());
//       lastSyncedText = updatedText.trim();
//       showToast("Text reset.");
//       setTimeout(() => {
//         textSyncPaused = false;
//       }, 500);
//     });
//   });

//   $("#backspaceText").click(function () {
//     console.log("[JS] Backspace button clicked");
//     textSyncPaused = true;
//     $.post("/backspace_text", function (updatedText) {
//       console.log("[JS] Backspace POST response:", updatedText);
//       $("#predictedText").val(updatedText.trim());
//       lastSyncedText = updatedText.trim();
//       showToast("Last word removed.");
//       setTimeout(() => {
//         textSyncPaused = false;
//       }, 500);
//     });
//   });

//   $("#addGestureForm").submit(function (e) {
//     e.preventDefault();
//     const word = $("#gestureInput").val();
//     if (!word.trim()) {
//       showToast("Please enter a valid gesture word.");
//       return;
//     }

//     $("#addGestureForm button").prop("disabled", true);

//     $.post("/capture", { word: word }, function (res) {
//       if (res.status === "exists") {
//         const userInput = prompt(`Gesture "${word}" already exists.\nPress R to overwrite or E to exit.`);
//         if (userInput && userInput.toUpperCase() === "R") {
//           $.post("/overwrite_capture", { word: word }, function () {
//             showToast(`Gesture "${word}" overwritten in both datasets.`);
//             $("#gestureInput").val("");
//             $("#addGestureForm button").prop("disabled", false);
//           });
//         } else {
//           showToast("Gesture capture canceled.");
//           $("#addGestureForm button").prop("disabled", false);
//         }
//       } else {
//         showToast(`Gesture "${word}" added to both datasets.`);
//         $("#gestureInput").val("");
//         $("#addGestureForm button").prop("disabled", false);
//       }
//     }).fail(function () {
//       showToast(`⚠️ Error: Failed to capture gesture "${word}".`);
//       $("#addGestureForm button").prop("disabled", false);
//     });
//   });

//   $("#downloadBtn").click(function () {
//     const mode = $("#modeSelector").val();
//     window.location.href = `/download_dataset?mode=${mode}`;
//   });

//   // SSE: Text stream
//   let eventSource;
//   function connectStream() {
//     try {
//       eventSource = new EventSource("/text_stream");

//       eventSource.onmessage = function (event) {
//         if (!textSyncPaused && event.data !== undefined) {
//           const newText = event.data.trim();
//           if (newText !== lastSyncedText) {
//             $("#predictedText").val(newText + " ");
//             lastSyncedText = newText;
//           }
//         }
//       };

//       eventSource.onerror = function (err) {
//         console.warn("[JS] SSE connection error. Retrying in 3s...", err);
//         eventSource.close();
//         setTimeout(connectStream, 3000);
//       };
//     } catch (e) {
//       console.error("[JS] SSE setup failed", e);
//       setTimeout(connectStream, 3000);
//     }
//   }

//   connectStream();
// });

// // Inject toast CSS
// $('<style>\
// .custom-toast {\
//   position: fixed;\
//   bottom: 20px;\
//   right: 20px;\
//   background: #222;\
//   color: #fff;\
//   padding: 12px 20px;\
//   border-radius: 8px;\
//   box-shadow: 0 0 10px rgba(0,0,0,0.2);\
//   z-index: 1000;\
//   font-weight: 600;\
//   opacity: 0.95;\
// }\
// </style>').appendTo("head");
