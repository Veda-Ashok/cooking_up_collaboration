// Persistent network connection that will be used to transmit real-time data
var socket = io();

var config;
var experimentParams = {
    layouts : ["cramped_room", "counter_circuit"],
    gameTime : 10,
    playerZero : "DummyAI"
};

var lobbyWaitTime = 300000;

/* * * * * * * * * * * * * 
 * Socket event handlers *
 * * * * * * * * * * * * */

window.intervalID = -1;
window.ellipses = -1;
window.lobbyTimeout = -1;

socket.on('waiting', function(data) {
    // Show game lobby
    $('#game-over').hide();
    $("#overcooked").empty();
    $('#lobby').show();
    if (!data.in_game) {
        if (window.intervalID === -1) {
            // Occassionally ping server to try and join
            window.intervalID = setInterval(function() {
                socket.emit('join', {});
            }, 1000);
        }
    }
    if (window.lobbyTimeout === -1) {
        // Waiting animation
        window.ellipses = setInterval(function () {
            var e = $("#ellipses").text();
            $("#ellipses").text(".".repeat((e.length + 1) % 10));
        }, 500);
        // Timeout to leave lobby if no-one is found
        window.lobbyTimeout = setTimeout(function() {
            socket.emit('leave', {});
        }, config.lobbyWaitTime)
    }
});

socket.on('creation_failed', function(data) {
    // Tell user what went wrong
    let err = data['error']
    $("#overcooked").empty();
    $('#overcooked').append(`<h4>Sorry, game creation code failed with error: ${JSON.stringify(err)}</>`);
    $("error-exit").show();

    // Let parent window (psiturk) know error occurred
    window.top.postMessage({ name : "error"}, "*");
});

socket.on('start_game', function(data) {
    // Hide game-over and lobby, show game title header
    if (window.intervalID !== -1) {
        clearInterval(window.intervalID);
        window.intervalID = -1;
    }
    if (window.lobbyTimeout !== -1) {
        clearInterval(window.ellipses);
        clearTimeout(window.lobbyTimeout);
        window.lobbyTimeout = -1;
        window.ellipses = -1;
    }
    graphics_config = {
        container_id : "overcooked",
        start_info : data.start_info
    };
    $("#overcooked").empty();
    $('#game-over').hide();
    $('#lobby').hide();
    $('#reset-game').hide();
    $('#game-title').show();
    enable_key_listener();
    enable_controller_listener();
    graphics_start(graphics_config);
});

socket.on('reset_game', function(data) {
    graphics_end();
    disable_key_listener();
    disable_controller_listener();
    $("#overcooked").empty();
    $("#reset-game").show();
    setTimeout(function() {
        $("#reset-game").hide();
        graphics_config = {
            container_id : "overcooked",
            start_info : data.state
        };
        graphics_start(graphics_config);
        enable_key_listener();
        enable_controller_listener();

        // Propogate game stats to parent window (psiturk)
        window.top.postMessage({ name : "data", data : data.data, done : false}, "*");
    }, data.timeout);
});

socket.on('state_pong', function(data) {
    // Draw state update
    drawState(data['state']);
});

socket.on('end_game', function(data) {
    // Hide game data and display game-over html
    graphics_end();
    disable_key_listener();
    disable_controller_listener();
    $('#game-title').hide();
    $('#game-over').show();
    $("#overcooked").empty();

    // Game ended unexpectedly
    if (data.status === 'inactive') {
        $("#error").show();
        $("#error-exit").show();
    }

    // Propogate game stats to parent window with psiturk code
    window.top.postMessage({ name : "data", data : data.data, done : true }, "*");
});

socket.on('end_lobby', function() {
    // Display join game timeout text
    $("#finding_partner").text(
        "We were unable to find you a partner."
    );
    $("#error-exit").show();

    // Stop trying to join
    clearInterval(window.intervalID);
    clearInterval(window.ellipses);
    window.intervalID = -1;

    // Let parent window (psiturk) know what happened
    window.top.postMessage({ name : "timeout" }, "*");
})


/* * * * * * * * * * * * * * 
 * Game Key Event Listener *
 * * * * * * * * * * * * * */

function enable_key_listener() {
    $(document).on('keydown', function(e) {
        let action = 'STAY'
        switch (e.which) {
            case 37: // left
                action = 'LEFT';
                break;

            case 38: // up
                action = 'UP';
                break;

            case 39: // right
                action = 'RIGHT';
                break;

            case 40: // down
                action = 'DOWN';
                break;

            case 32: //space
                action = 'SPACE';
                break;

            default: // exit this handler for other keys
                return; 
        }
        e.preventDefault();
        socket.emit('action', { 'action' : action });
    });
};

function disable_key_listener() {
    $(document).off('keydown');
};


/* * * * * * * * * * * * * * * * * * 
 * Game Controller Event Listener *
 * * * * * * * * * * * * * * * * * */

window.gamepadState = {
    lastAction: null,
    lastButtonState: {},
    pollInterval: null,
    deadzone: 0.3  // Analog stick deadzone threshold
};

function enable_controller_listener() {
    // Start polling for gamepad input
    window.gamepadState.pollInterval = setInterval(pollGamepad, 100); // Poll every 100ms
}

function disable_controller_listener() {
    // Stop polling for gamepad input
    if (window.gamepadState.pollInterval) {
        clearInterval(window.gamepadState.pollInterval);
        window.gamepadState.pollInterval = null;
    }
    window.gamepadState.lastAction = null;
    window.gamepadState.lastButtonState = {};
}

function pollGamepad() {
    const gamepads = navigator.getGamepads();
    if (!gamepads || gamepads.length === 0) return;
    
    // Use the first connected gamepad
    const gamepad = gamepads[0];
    if (!gamepad) return;
    
    let action = null;
    
    // Check D-pad buttons (buttons 12-15 on standard gamepad)
    if (gamepad.buttons[12] && gamepad.buttons[12].pressed) {
        action = 'UP';
    } else if (gamepad.buttons[13] && gamepad.buttons[13].pressed) {
        action = 'DOWN';
    } else if (gamepad.buttons[14] && gamepad.buttons[14].pressed) {
        action = 'LEFT';
    } else if (gamepad.buttons[15] && gamepad.buttons[15].pressed) {
        action = 'RIGHT';
    }
    
    // Check left analog stick (axes 0 and 1)
    if (!action && gamepad.axes.length >= 2) {
        const xAxis = gamepad.axes[0];
        const yAxis = gamepad.axes[1];
        
        // Apply deadzone
        if (Math.abs(xAxis) > window.gamepadState.deadzone || Math.abs(yAxis) > window.gamepadState.deadzone) {
            // Determine dominant direction
            if (Math.abs(xAxis) > Math.abs(yAxis)) {
                action = xAxis > 0 ? 'RIGHT' : 'LEFT';
            } else {
                action = yAxis > 0 ? 'DOWN' : 'UP';
            }
        }
    }
    
    // Check B button (button 1 on standard gamepad) for interact/pickup
    if (gamepad.buttons[1] && gamepad.buttons[1].pressed) {
        // Only send action if button wasn't pressed in last poll (prevent spam)
        if (!window.gamepadState.lastButtonState[1]) {
            socket.emit('action', { 'action' : 'SPACE' });
            window.gamepadState.lastButtonState[1] = true;
        }
    } else {
        window.gamepadState.lastButtonState[1] = false;
    }
    
    // Send movement action if it changed
    if (action && action !== window.gamepadState.lastAction) {
        socket.emit('action', { 'action' : action });
        window.gamepadState.lastAction = action;
    } else if (!action && window.gamepadState.lastAction) {
        // Reset when no direction is pressed
        window.gamepadState.lastAction = null;
    }
}


/* * * * * * * * * * * * 
 * Game Initialization *
 * * * * * * * * * * * */

socket.on("connect", function() {
    // set configuration variables
    set_config();

    // Config for this specific game
    let uid = $('#uid').text();
    let params = JSON.parse(JSON.stringify(config.experimentParams));
    params.psiturk_uid = uid;
    let data = {
        "params" : params,
        "game_name" : "psiturk"
    };

    // create (or join if it exists) new game
    socket.emit("join", data);
});


/* * * * * * * * * * *
 * Utility Functions *
 * * * * * * * * * * */

var arrToJSON = function(arr) {
    let retval = {}
    for (let i = 0; i < arr.length; i++) {
        elem = arr[i];
        key = elem['name'];
        value = elem['value'];
        retval[key] = value;
    }
    return retval;
};

var set_config = function() {
    config = JSON.parse($("#config").text());
}
