<?php

use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateTableMfSkfdDerivereport extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('mf_skfd_derivereport', function($table) {
	    $table->increments('id');
	    $table->string('fd_code')->comment('基金代码');
	    $table->date('date')->default('0000-00-00')->comment('行情日期');
	    $table->decimal('sharp_y',10,6);
            $table->decimal('sharp_2y',10,6);
            $table->decimal('sharp_5y',10,6);
            $table->decimal('sortino_y',10,6);
            $table->decimal('sortino_2y',10,6);
            $table->decimal('sortino_5y',10,6);
            $table->decimal('jenson_y',10,6);
            $table->decimal('jenson_2y',10,6);
            $table->decimal('jenson_5y',10,6);
	    $table->timestamps();
	});
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::drop('mf_skfd_derivereport');
    }
}
