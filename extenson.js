const vscode = require('vscode');
const net = require('net');

let client;

function activate(context) {
    let disposable = vscode.commands.registerCommand('ai-linter.startLinting', function () {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            const document = editor.document;
            const code = document.getText();

            // Connect to the socket server
            client = new net.Socket();
            client.connect(65432, '127.0.0.1', function () {
                console.log('Connected to server');
                client.write(code);
            });

            // Receive linting results and display in the editor
            client.on('data', function (data) {
                vscode.window.showInformationMessage('Linting Suggestions: ' + data.toString());
                client.destroy();  // Close connection after receiving data
            });

            client.on('close', function () {
                console.log('Connection closed');
            });
        }
    });

    context.subscriptions.push(disposable);
}

function deactivate() {
    if (client) {
        client.destroy();
    }
}

module.exports = {
    activate,
    deactivate
};
