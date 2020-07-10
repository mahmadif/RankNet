# Implements the well-known pairwise-based learning to rank algorithm proposed by Burges et al., ICML-2005

# input
	# X_train: matrix of training instances (one instance per row)
	# Y_train: ranking of the instances in the training data
	# net_params: list of network parameters:
		# n_hidden, n_units, batch_normalization, epochs, batch_size, callbacks, validation_split, optimizer_lr, regularizer_l2_l,
		# layer_activation, output_activation, verbose
# output
	# fitted_model: fitted network
	# prediction_model: the model that can further be used for prediction
	
func_RankNet <- function( X_train, Y_train, net_params ){

	f_construct_prediction_network <- function(){
	
		xx <- inp <- layer_input(shape = n_features, name = "")
		
		output_node1 <- layer_dense(
			units = 1, 
			activation = net_params$output_activation, 
			kernel_regularizer = regularizer_l2(l = net_params$regularizer_l2_l), 
			name = "output_node",
			weights = get_weights(get_layer(object = Net, name = "output_node"))
			)
		
		for(i in seq(n_hidden)){
		
			nm <- ifelse(batch_normalization, paste0("hiddenNormalized_", i), paste0("hidden_", i))
			
			dense <- layer_dense(
				units = n_units, 
				kernel_regularizer = regularizer_l2(l = net_params$regularizer_l2_l), 
				activation = net_params$layer_activation, 
				name =  nm,
				weights = get_weights(get_layer(object = Net, name = nm))
				)
				
			if (batch_normalization==TRUE){
				activation_ <- layer_activation(activation = net_params$layer_activation)
				batchnorm <- layer_batch_normalization( weights = get_weights(get_layer(object = Net, name = paste0("batch_normalization_", i))) )
				xx <- batchnorm( activation_( dense(xx) ) )
			} else {
				xx <- dense(xx)
			}
		}

		output_score <- output_node1(xx)

		keras_model(inputs = c(inp), outputs = output_score)
	}
	f_constructNetwork <- function(){
		
		hidden_layers <- list()
		# ===== Construct_layers ==========
		in1 <- layer_input(shape = c(n_features), name = "IN1")
		in2 <- layer_input(shape = c(n_features), name = "IN2")
		
		output_node <- layer_dense(
			units = 1, 
			activation = net_params$output_activation, 
			kernel_regularizer = regularizer_l2(l = net_params$regularizer_l2_l), 
			name = "output_node"
			)
		
		for(i in seq(n_hidden)){
			dense <- layer_dense(
				units = n_units, 
				kernel_regularizer = regularizer_l2(l = net_params$regularizer_l2_l), 
				activation = net_params$layer_activation, 
				name =  ifelse(batch_normalization, paste0("hiddenNormalized_", i), paste0("hidden_", i)) 
				)
			hidden_layers <- c(hidden_layers, dense)
		}

		# =================================
		if (batch_normalization==TRUE){

			for( i in seq(n_hidden) ){
			
				if (i==1){
					activation_ <- layer_activation(activation = net_params$layer_activation)
					batchnorm <- layer_batch_normalization( name=paste0("batch_normalization_", i) )
					enc_x1 <- batchnorm( activation_( hidden_layers[[i]](in1) ) )
					
					enc_x2 <- batchnorm( activation_( hidden_layers[[i]](in2) ) )
					neg_x2 <- layer_lambda(object = enc_x2, f = function(x) -x, name="negation_func")
					
					next
				}
			
				activation_ <- layer_activation(activation = net_params$layer_activation)
				batchnorm <- layer_batch_normalization( name=paste0("batch_normalization_", i) )
				enc_x1 <- batchnorm( activation_( hidden_layers[[i]](enc_x1) ) )
				
				neg_x2 <- batchnorm( activation_( hidden_layers[[i]](neg_x2) ) )
			}
			
		} else { # batch_normalization = FALSE
			
			for( i in seq(n_hidden) ){
			
				if (i==1){
					enc_x1 <- hidden_layers[[i]](in1)
					enc_x2 <- hidden_layers[[i]](in2)

					neg_x2 <- layer_lambda(object = enc_x2, f = function(x) -x, name="negation_func")
					
					next
				}
			
				enc_x1 <- hidden_layers[[i]](enc_x1)
				neg_x2 <- hidden_layers[[i]](neg_x2)
			}
		}

		merged_inputs <- layer_add(inputs = list(enc_x1, neg_x2), name="diff_score")

		output <- output_node(merged_inputs)

		model <- keras_model(inputs = c(in1, in2), outputs = output)
		
		model
	}
	
	library(keras)
	n_features <- ncol(X_train)
	n_hidden <- net_params$n_hidden
	batch_normalization <- net_params$batch_normalization
	n_units <- net_params$n_units
	
	(X_sorted <- X_train[ order(Y_train) ,])
	cmb <- combn(nrow(X_sorted),2)

	(sq <- seq(1, ncol(cmb), 2))
	tmp <- cmb[,sq]
	cmb[,sq][1,] <- tmp[2,]
	cmb[,sq][2,] <- tmp[1,]

	Pairs <- X_sorted[cmb,]
	in1 <- Pairs[ c(TRUE, FALSE) ,]
	in2 <- Pairs[ c(FALSE, TRUE) ,]
	target <- rep_len(c(0,1), length.out = nrow(in1) )
	
	valid_set <- ceiling(nrow(in1)*net_params$validation_split)
	cat("Train on <", nrow(in1)-valid_set,"> samples, validate on <", valid_set ,"> samples\n")
	
	Net <- f_constructNetwork()
	
	print("<Compiling model started>")
	Net %>% compile(
		loss = 'binary_crossentropy',
		optimizer = optimizer_sgd(lr = net_params$optimizer_lr),
		# optimizer = optimizer_adam(),
		metrics = c('accuracy')
	)
	
	print("<Fitting model started>")
	Net %>% fit(
		list(in1, in2), 
		target, 
		epochs = net_params$epochs, 
		batch_size = net_params$batch_size, 
		callbacks = net_params$callbacks, 
		validation_split = net_params$validation_split, 
		verbose=net_params$verbose,
		shuffle = TRUE
		)
	
	list( 
		fitted_model = Net,
		prediction_model = f_construct_prediction_network() 
		)
}
