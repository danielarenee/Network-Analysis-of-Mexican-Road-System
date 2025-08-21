
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAXSIZE 100

// DEFINICIÓN DE PRODUCTO
typedef struct {
    int id;            
    char name[50];     
    int items;         
    float cost;        
    char date[15];     
    int priority;      //Características del producto
} Product;

// Declaración de variables globales
Product *heap = NULL;  // Puntero al heap (memoria dinámica)
int count = 0;         // Número actual de productos en el heap
int maxSize = 0;       // Tamaño máximo del heap

// PROTOTIPOS DE FUNCIONES AUXILIARES
void insertProduct(Product product);
void loadFromFile(char *archivoProductos);
void insertByKeyboard();
void orderByPriority();
Product removeTop();

// FUNCIONES PARA MANEJAR EL HEAP
// Reorganizar el heap hacia arriba
void reHeapUp(int index) {
    int parent = (index - 1) / 2;
    if (index > 0 && heap[index].priority < heap[parent].priority) {
        // Intercambiar si la prioridad del hijo es menor que la del padre
        Product temp = heap[index];
        heap[index] = heap[parent];
        heap[parent] = temp;
        reHeapUp(parent); // Llamada recursiva
    }
}

// Reorganizar el heap hacia abajo
void reHeapDown(Product heap[], int count, int index, char *criterion) {
    int leftChild = 2 * index + 1;
    int rightChild = 2 * index + 2;
    int smallest = index;

    // Decidir el criterio de reorganización
    if (strcmp(criterion, "priority") == 0) {
        if (leftChild < count && heap[leftChild].priority < heap[smallest].priority) {
            smallest = leftChild;
        }
        if (rightChild < count && heap[rightChild].priority < heap[smallest].priority) {
            smallest = rightChild;
        }
    } else if (strcmp(criterion, "cost") == 0) {
        // Comparar por costo
        if (leftChild < count && heap[leftChild].cost < heap[smallest].cost) {
            smallest = leftChild;
        }
        if (rightChild < count && heap[rightChild].cost < heap[smallest].cost) {
            smallest = rightChild;
        }
    } else if (strcmp(criterion, "name") == 0) {
        // Comparar alfabéticamente por nombre
        if (leftChild < count && strcmp(heap[leftChild].name, heap[smallest].name) < 0) {
            smallest = leftChild;
        }
        if (rightChild < count && strcmp(heap[rightChild].name, heap[smallest].name) < 0) {
            smallest = rightChild;
        }
    }

    // Si es necesario un intercambio
    if (smallest != index) {
        Product temp = heap[index];
        heap[index] = heap[smallest];
        heap[smallest] = temp;
        reHeapDown(heap, count, smallest, criterion); // Llamada recursiva
    }
}

// Remover el elemento con mayor prioridad (raíz del heap)
Product removeTop() {
    if (count == 0) {
        // Heap vacío, devolver un producto vacío
        Product empty = {-1, "", 0, 0.0, "", -1};
        return empty;
    }

    // Guardar el elemento raíz y reorganizar el heap
    Product top = heap[0];
    heap[0] = heap[count - 1];
    count--; // Reducir el tamaño del heap
    reHeapDown(heap, count, 0, "priority");
    return top;
}

// FUNCIONES PARA INSERCIÓN
// Verificar si un producto ya existe en el heap
int productExists(char *name) {
    for (int i = 0; i < count; i++) {
        if (strcmp(heap[i].name, name) == 0) {
            return 1; // Producto encontrado
        }
    }
    return 0; // Producto no encontrado
}

//INSERTAR ---------------------------------------------
//INSERTAR (GENERAL)
void insertProduct(Product product) {
    if (count >= maxSize) {
        printf("\nNo se puede insertar más productos, el heap está lleno.\n");
        return;
    }
    if (productExists(product.name)) {
        printf("\nYa existe un producto con el nombre '%s'. No se puede insertar duplicado.\n", product.name);
        return;
    }

    // Insertar el producto en el heap y reorganizar
    heap[count] = product;
    reHeapUp(count); // Reorganizar hacia arriba
    count++; // Incrementar el número de productos
    printf("\nProducto '%s' insertado exitosamente.\n", product.name);
}

// Cargar productos desde un archivo
void loadFromFile(char *archivoProductos) {
    FILE *fp = fopen(archivoProductos, "r");
    if (fp == NULL) {
        printf("\nNo se pudo abrir el archivo '%s'\n", archivoProductos);
        return;
    }

    Product temp;
    while (fscanf(fp, "%d %[^\n] %d %f %s %d", 
                  &temp.id, temp.name, &temp.items, &temp.cost, temp.date, &temp.priority) == 6) {
        // Insertar solo si el producto no existe
        if (!productExists(temp.name)) {
            insertProduct(temp);
        } else {
            printf("\nProducto con nombre '%s' ya existe. No se insertó desde el archivo.\n", temp.name);
        }
    }

    fclose(fp);
    printf("\nArchivo '%s' cargado exitosamente.\n", archivoProductos);
}

// Insertar producto desde teclado
void insertByKeyboard() {
    Product item;

    printf("Ingrese el ID del producto: ");
    while (scanf("%d", &item.id) != 1 || item.id <= 0) {
        printf("ID inválido, por favor ingrese un número positivo: ");
        while (getchar() != '\n'); // Limpiar el buffer
    }

    printf("Ingrese el nombre del producto: ");
    getchar(); // Limpiar el buffer de entrada
    fgets(item.name, sizeof(item.name), stdin);
    item.name[strcspn(item.name, "\n")] = '\0'; // Eliminar el salto de línea

    printf("Ingrese el número de artículos: ");
    while (scanf("%d", &item.items) != 1 || item.items < 0) {
        printf("Número de artículos inválido, por favor ingrese un valor no negativo: ");
        while (getchar() != '\n');
    }

    printf("Ingrese el costo: ");
    while (scanf("%f", &item.cost) != 1 || item.cost < 0) {
        printf("Costo inválido, por favor ingrese un valor no negativo: ");
        while (getchar() != '\n');
    }

    printf("Ingrese la fecha de ingreso (DD/MM/AAAA): ");
    scanf("%s", item.date);

    printf("Ingrese el nivel de prioridad: ");
    while (scanf("%d", &item.priority) != 1 || item.priority < 0) {
        printf("Prioridad inválida, por favor ingrese un valor no negativo: ");
        while (getchar() != '\n');
    }

    insertProduct(item);
}



// ELIMINAR --------------------------------------------

// POR NOMBRE
// Elimina un producto del heap buscando por su nombre
void deleteByName(char *name) {
    int found = -1;

    // Buscar el producto en el heap por nombre
    for (int i = 0; i < count; i++) {
        if (strcmp(heap[i].name, name) == 0) {
            found = i;
            break;
        }
    }

    // Si no se encuentra, se notifica al usuario
    if (found == -1) {
        printf("\nProducto no encontrado.\n");
        return;
    }

    // Sustituir el producto encontrado con el último del heap y reducir el tamaño
    heap[found] = heap[count - 1];
    count--;

    // Reorganizar el heap para mantener la propiedad de heap
    reHeapDown(heap, count, found, "priority");  

    // Confirmación de eliminación
    printf("\nProducto con nombre '%s' eliminado.\n", name);
}

// POR ID
// Elimina un producto del heap buscando por su ID
void deleteByID(int id) {
    int found = -1;

    // Buscar el producto en el heap por ID
    for (int i = 0; i < count; i++) {
        if (heap[i].id == id) {
            found = i;
            break;
        }
    }

    // Si no se encuentra, se notifica al usuario
    if (found == -1) {
        printf("\nProducto no encontrado.\n");
        return;
    }

    // Almacenar el nombre del producto para mostrarlo tras eliminarlo
    char productName[50];
    strcpy(productName, heap[found].name);

    // Sustituir el producto encontrado con el último del heap y reducir el tamaño
    heap[found] = heap[count - 1];
    count--;

    // Reorganizar el heap para mantener la propiedad de heap
    reHeapDown(heap, count, found, "priority"); 

    // Confirmación de eliminación
    printf("\nProducto '%s' con ID %d eliminado.\n", productName, id);
}

// PRIMERO
// Elimina el primer producto del heap (la raíz)
void deleteFirst() {
    // Verificar si el heap está vacío
    if (count == 0) {
        printf("\nEl heap está vacío.\n");
        return;
    }

    // Almacenar el nombre del producto para mostrarlo tras eliminarlo
    char productName[50];
    strcpy(productName, heap[0].name);

    // Llamar a una función que elimina la raíz del heap
    removeTop(); 

    // Confirmación de eliminación
    printf("\nPrimer producto '%s' eliminado.\n", productName);
}

// ULTIMO
// Elimina el último producto del heap
void deleteLast() {
    // Verificar si el heap está vacío
    if (count == 0) {
        printf("\nEl heap está vacío.\n");
        return;
    }

    // Reducir el tamaño del heap para "eliminar" el último elemento
    count--;

    // Confirmación de eliminación
    printf("\nÚltimo producto eliminado.\n");
}

// ACTUALIZAR --------------------------------------------

// COSTO
// Actualiza el costo de un producto en el heap buscando por su ID
void updateCost(int id, float newCost) {
    int found = -1;

    // Buscar el producto en el heap por ID
    for (int i = 0; i < count; i++) {
        if (heap[i].id == id) {
            found = i;
            break;
        }
    }

    // Si no se encuentra, se notifica al usuario
    if (found == -1) {
        printf("\nProducto con ID %d no encontrado.\n", id);
        return;
    }

    // Actualizar el costo del producto
    heap[found].cost = newCost;

    // Confirmación de actualización
    printf("\nCosto del producto con ID %d actualizado a %.2f.\n", id, newCost);
}

// NUM DE ITEMS
// Actualiza el número de artículos disponibles de un producto en el heap
void updateItems(int id, int newItems) {
    int found = -1;

    // Buscar el producto en el heap por ID
    for (int i = 0; i < count; i++) {
        if (heap[i].id == id) {
            found = i;
            break;
        }
    }

    // Si no se encuentra, se notifica al usuario
    if (found == -1) {
        printf("\nProducto con ID %d no encontrado.\n", id);
        return;
    }

    // Actualizar el número de artículos
    heap[found].items = newItems;

    // Confirmación de actualización
    printf("\nNúmero de artículos del producto con ID %d actualizado a %d.\n", id, newItems);
}

// FECHA
// Actualiza la fecha de entrada de un producto en el heap
void updateDate(int id, char *newDate) {
    int found = -1;

    // Buscar el producto en el heap por ID
    for (int i = 0; i < count; i++) {
        if (heap[i].id == id) {
            found = i;
            break;
        }
    }

    // Si no se encuentra, se notifica al usuario
    if (found == -1) {
        printf("\nProducto con ID %d no encontrado.\n", id);
        return;
    }

    // Actualizar la fecha de entrada
    strcpy(heap[found].date, newDate);

    // Confirmación de actualización
    printf("\nFecha de entrada del producto con ID %d actualizada a %s.\n", id, newDate);
}

// PRIORIDAD
// Actualiza la prioridad de un producto en el heap
void updatePriority(int id, int newPriority) {
    int found = -1;

    // Buscar el producto en el heap por ID
    for (int i = 0; i < count; i++) {
        if (heap[i].id == id) {
            found = i;
            break;
        }
    }

    // Si no se encuentra, se notifica al usuario
    if (found == -1) {
        printf("\nProducto con ID %d no encontrado.\n", id);
        return;
    }

    // Guardar la prioridad anterior
    int oldPriority = heap[found].priority;

    // Actualizar la prioridad
    heap[found].priority = newPriority;

    // Reorganizar el heap según sea necesario
    if (newPriority < oldPriority) {
        reHeapUp(found); // Si la nueva prioridad es más alta
    } else {
        reHeapDown(heap, count, found, "priority"); // Si la nueva prioridad es más baja
    }

    // Confirmación de actualización
    printf("\nPrioridad del producto con ID %d actualizada a %d.\n", id, newPriority);
}



//BUSCAR --------------------------------------------
//POR NOMBRE
void searchByName() {
    char name[50];
    printf("Ingrese el nombre del producto a buscar: ");
    getchar();
    fgets(name, sizeof(name), stdin);
    name[strcspn(name, "\n")] = '\0';
    //Se busca alguna coincidencia en el nombre dado

    printf("\nResultados de búsqueda:\n");
    for (int i = 0; i < count; i++) {
        if (strcmp(heap[i].name, name) == 0) { //si se encuentra la conincidencia se imprimen todas las características del producto
            printf("ID: %d, Nombre: %s, Artículos: %d, Costo: %.2f, Fecha: %s, Prioridad: %d\n",
                   heap[i].id, heap[i].name, heap[i].items, heap[i].cost, heap[i].date, heap[i].priority);
        }
    }
}

//POR PRECIO
void searchByPrice() {
    float exactPrice;
    printf("Ingrese el precio exacto: ");
    scanf("%f", &exactPrice);
    //Se recive el precio y se busca la coincidencia
    printf("\nResultados de búsqueda:\n");
    int found = 0;
    //Si encuentra en el arreglo de los productos, imprime todas las caracteristicas del producto
    for (int i = 0; i < count; i++) {
        if (heap[i].cost == exactPrice) {
            printf("ID: %d | %s (%d)\n",
                   heap[i].id, heap[i].name, heap[i].items);
            found = 1;
        }
    }

    if (!found) {
        printf("No se encontraron resultados con el precio exacto %.2f\n", exactPrice);
    }
    //Si no se encuentra algun producto se imprime un mensaje de error
}

//POR FECHA DE ENTRADA
void searchByDate() {
    char date[15];
    printf("Ingrese la fecha (DD/MM/AAAA): ");
    scanf("%s", date);
    //De nuevo se vuelve a buscar en el arreglo alguna coincidencia de fechas
    printf("\nResultados de búsqueda:\n");
    for (int i = 0; i < count; i++) {
        if (strcmp(heap[i].date, date) == 0) 
        {
            printf("ID: %d | %s (%d)\n",
                   heap[i].id, heap[i].name, heap[i].items);
        }
    }
}


//MOSTRAR --------------------------------------------
//POR PRIORIDAD
void showHeapOrdered() {

    //Si la cuenta es igual a 0 es porque no existen productos dentro del heap
    if (count == 0) {
        printf("\nEl heap está vacío.\n");
        return;
    }

    Product tempHeap[MAXSIZE];
    int tempCount = 0;

    // Copiar elementos del heap original y crear un nuevo heap para asegurarse del orden adecuado
    for (int i = 0; i < count; i++) {
        tempHeap[tempCount] = heap[i];
        reHeapUp(tempCount);
        tempCount++;
    }

    printf("\n%-5s %-20s %-10s %-10s %-12s %-10s", "ID", "Nombre", "Artículos", "Costo", "Fecha", "Prioridad");
    printf("\n----------------------------------------------------------------------------");

    // Extraer los elementos ordenados desde la raíz del heap temporal
    while (tempCount > 0) {
        Product top = tempHeap[0];
        printf("\n%-5d %-20s %-10d %-10.2f %-12s %-10d",
               top.id, top.name, top.items, top.cost, top.date, top.priority);

        tempHeap[0] = tempHeap[tempCount - 1]; // Reemplaza la raíz por el último elemento
        tempCount--;
        reHeapDown(tempHeap, tempCount, 0, "priority");   // Ajusta el heap después de la extracción
    }
    printf("\n");
}

//POR PRECIO
void showHeapOrderedByPrice() {

    //Otra vez, si el count es 0 el heap está vacio
    if (count == 0) {
        printf("\nEl heap está vacío.\n");
        return;
    }

    Product tempHeap[MAXSIZE];
    int tempCount = count;

    // Copiar elementos del heap original
    for (int i = 0; i < count; i++) {
        tempHeap[i] = heap[i];
    }

    // Reorganizar el heap temporal basado en el precio
    for (int i = (tempCount / 2) - 1; i >= 0; i--) {
        reHeapDown(tempHeap, tempCount, i, "cost");
    }

    printf("\n%-5s %-20s %-10s %-10s %-12s %-10s", "ID", "Nombre", "Artículos", "Costo", "Fecha", "Prioridad");
    printf("\n----------------------------------------------------------------------------");

    // Extraer los elementos ordenados desde la raíz del heap temporal
    while (tempCount > 0) {
        Product top = tempHeap[0];
        printf("\n%-5d %-20s %-10d %-10.2f %-12s %-10d",
               top.id, top.name, top.items, top.cost, top.date, top.priority);

        tempHeap[0] = tempHeap[tempCount - 1];
        tempCount--;
        reHeapDown(tempHeap, tempCount, 0, "cost");   // Ajustar el heap basado en costo
    }
    printf("\n");
}

//POR NOMBRE 
void showHeapOrderedByName() {
    if (count == 0) {
        printf("\nEl heap está vacío.\n");
        return;
    }
    //Si el heap no tiene elementos, marca error

    Product tempHeap[MAXSIZE];
    int tempCount = count;

    // Copiar elementos del heap original
    for (int i = 0; i < count; i++) {
        tempHeap[i] = heap[i];
    }

    // Reorganizar el heap temporal basado en el nombre
    for (int i = (tempCount / 2) - 1; i >= 0; i--) {
        reHeapDown(tempHeap, tempCount, i, "name");
    }

    // Mostrar encabezado
    printf("\n%-5s %-20s %-10s %-10s %-12s %-10s", "ID", "Nombre", "Artículos", "Costo", "Fecha", "Prioridad");
    printf("\n----------------------------------------------------------------------------");

    // Extraer los elementos ordenados desde la raíz del heap temporal
    while (tempCount > 0) {
        Product top = tempHeap[0];
        printf("\n%-5d %-20s %-10d %-10.2f %-12s %-10d",
               top.id, top.name, top.items, top.cost, top.date, top.priority);

        tempHeap[0] = tempHeap[tempCount - 1];
        tempCount--;
        reHeapDown(tempHeap, tempCount, 0, "name");   // Ajusta el heap después de la extracción basado en nombre
    }
    printf("\n");
}


// MENU PRINCIPAL ---------------------------------------
void menu() {
    int option, op, IDprod, id;
    char name[50];

    Product item;

    // Menú principal: permite gestionar el inventario de productos
    do {
        printf("\n---| I N V E N T A R I O - D E - S U P E R M E R C A D O |---\n");
        printf("\n1) Insertar producto");
        printf("\n2) Eliminar producto");
        printf("\n3) Actualizar producto");
        printf("\n4) Buscar producto");
        printf("\n5) Mostrar productos");
        printf("\n6) Salir");
        printf("\n\nElija una opción: ");
        scanf("%d", &option);

        // Procesa la opción seleccionada mediante un switch
        switch (option) {

            case 1: // Inserta un producto al inventario
                printf("\n---| I N S E R T A R - P R O D U C T O |---\n");
                printf("\n1) Insertar manualmente");
                printf("\n2) Buscar archivo");
                printf("\n3) Regresar");
                printf("\nElija una opción: ");
                scanf("%d", &op);

                if (op == 1) { 
                    insertByKeyboard(); // Inserta producto ingresando datos manualmente
                } else if (op == 2) { 
                    char archivoProductos[100];
                    printf("\nEscribe el nombre del archivo: ");
                    scanf("%s", archivoProductos);
                    loadFromFile(archivoProductos); // Carga productos desde un archivo externo
                } else if (op != 3) {
                    printf("\nOpción inválida. Intente nuevamente.\n");
                }
                break;
        
            case 2: // Elimina un producto del inventario
                printf("\n---| E L I M I N A R - P R O D U C T O |---\n");
                printf("\n1) Eliminar por nombre");
                printf("\n2) Eliminar por ID");
                printf("\n3) Eliminar el primer producto");
                printf("\n4) Eliminar el último producto");
                printf("\n5) Regresar");
                printf("\nElija una opción: ");
                scanf("%d", &op);

                if (op == 1) {
                    printf("\nEscribe el nombre del producto a eliminar: ");
                    getchar();  
                    fgets(name, sizeof(name), stdin);
                    name[strcspn(name, "\n")] = '\0'; 
                    deleteByName(name); // Elimina un producto buscando por su nombre
                } else if (op == 2) {
                    printf("\nEscribe el ID del producto a eliminar: ");
                    scanf("%d", &id);
                    deleteByID(id); // Elimina un producto buscando por su ID             
                } else if (op == 3) {
                    deleteFirst(); // Elimina el primer producto del inventario
                } else if (op == 4) {
                    deleteLast(); // Elimina el último producto del inventario
                } else if (op != 5) {
                    printf("\nOpción inválida. Intente nuevamente.\n");
                }
                break;

            case 3: // Actualiza la información de un producto
                printf("\n---| A C T U A L I Z A R - P R O D U C T O |---\n");
                printf("\n Ingresa el ID del producto a actualizar: ");
                scanf("%d", &IDprod);

                printf("\n1) Actualizar costo");
                printf("\n2) Actualizar número de artículos");
                printf("\n3) Actualizar fecha de entrada");
                printf("\n4) Actualizar prioridad");
                printf("\n5) Regresar");
                printf("\n\nElija una opción: ");
                scanf("%d", &op);

                if (op == 1) {
                    float newCost;
                    printf("\nIngrese el nuevo costo: ");
                    scanf("%f", &newCost);
                    updateCost(IDprod, newCost); // Modifica el costo del producto
                } else if (op == 2) {
                    int newItems;
                    printf("\nIngrese el nuevo número de artículos: ");
                    scanf("%d", &newItems);
                    updateItems(IDprod, newItems); // Modifica el stock disponible
                } else if (op == 3) {
                    char newDate[15];
                    printf("\nIngrese la nueva fecha de entrada (DD/MM/AAAA): ");
                    scanf("%s", newDate);
                    updateDate(IDprod, newDate); // Modifica la fecha de ingreso
                } else if (op == 4) {
                    int newPriority;
                    printf("\nIngrese la nueva prioridad: ");
                    scanf("%d", &newPriority);
                    updatePriority(IDprod, newPriority); // Cambia el nivel de prioridad
                } else if (op != 5) {
                    printf("\nOpción inválida. Intente nuevamente.\n");
                }
                break;

            case 4: // Busca un producto según diferentes criterios
                printf("\n---| B U S C A R - P R O D U C T O |---\n");
                printf("\n1) Buscar por nombre");
                printf("\n2) Buscar por precio");
                printf("\n3) Buscar por fecha de entrada");
                printf("\n4) Regresar");
                printf("\nElija una opción: ");
                scanf("%d", &op);

                if (op == 1) {
                    searchByName(); // Muestra datos de un producto por su nombre
                } else if (op == 2) {
                    searchByPrice(); // Lista productos que coincidan con un precio
                } else if (op == 3) {
                    searchByDate(); // Lista productos que coincidan con una fecha
                } else if (op != 4) {
                    printf("\nOpción inválida. Intente nuevamente.\n");
                }   
                break;

            case 5: // Muestra todos los productos según un criterio
                printf("\n---| M O S T R A R - P R O D U C T O S |---\n");
                printf("\n1) Mostrar por prioridad");
                printf("\n2) Mostrar por precio");
                printf("\n3) Mostrar por nombre");
                printf("\n4) Regresar");
                printf("\nElija una opción: ");
                scanf("%d", &op);

                if (op == 1) 
                {
                    showHeapOrdered(); // Lista productos ordenados por prioridad
                } else if (op == 2) 
                {
                    showHeapOrderedByPrice(); // Lista productos ordenados por precio
                } else if (op == 3) 
                {
                    showHeapOrderedByName(); // Lista productos ordenados por nombre
                } else if (op != 4) 
                {
                    printf("\nOpción inválida. Intente nuevamente.\n");
                }  
                break;

            case 6: // Finaliza el programa
                printf("\nSaliendo del programa...\n");
                break;

            default: // Muestra mensaje de error si la opción es inválida
                printf("\nOpción inválida. Intente nuevamente.\n");
        }
    } while (option != 6); // Repite mientras no elija salir
}

int main() 
{
    // Solicita al usuario el tamaño máximo del inventario
    printf("Ingrese el número máximo de productos del inventario: ");
    scanf("%d", &maxSize);

    // Asigna memoria dinámica para el heap (inventario)
    heap = (Product *)malloc(maxSize * sizeof(Product));
    if (heap == NULL) {
        printf("Error al asignar memoria.\n");
        return 1; // Termina el programa si no se pudo asignar memoria
    }

    menu(); // Llama al menú principal

    free(heap); // Libera la memoria asignada para el heap
    return 0; // Fin del programa
}

