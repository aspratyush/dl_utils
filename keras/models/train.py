from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
from keras.optimizers import SGD

#model, X, Y_oh, x_ph=x_ph, y_ph=y_ph, sess=sess, 
#            model_path='./saved_model/lenet_dogscats',nb_epochs=NB_EPOCHS, 
#            nb_batches=NB_BATCHES, nb_rows=NB_ROWS, nb_cols=NB_COLS, 
#            nb_channels=NB_CHANNELS, nb_classes=NB_CLASSES)

def run(model, X, Y, optimizer=None, nb_epochs=30, nb_batches=128):
    
    if optimizer==None:
        optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    # compile the model
    print('Model compile...')
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
   

    # run the training
    print('Model fit...')
    print(X['train'].shape, Y['train'].shape)
    history = model.fit(X['train'],
            Y['train'],
            epochs=nb_epochs,
            batch_size=nb_batches,
            validation_data=(X['valid'], Y['valid']))

    # Evaluate the model on test data
    score = model.evaluate(X['test'], Y['test'], batch_size=nb_batches)
    print('Score : ', score)

    return history, score
