run : 
	docker-compose up -d

delete :
	docker-compose down

.PHONY: run delete